use crate::arrow::array_reader::{ArrayReader, RowGroups};
use crate::arrow::arrow_reader::RowSelector;
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::Encoding;
use crate::column::page::{Page, PageIterator, PageReader};
use crate::column::writer::encoder::ColumnValues;
use crate::errors::ParquetError;
use crate::file::reader::{Length, RowGroupReader};
use crate::schema::types::ColumnDescPtr;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType as ArrowType, Field, Schema};
use bytes::Bytes;
use ignition::bundle::{IgnitionBundle, MappedFd};
use ignition::config::ConfigBuilder;
use ignition::{IgnitionJob, IgnitionRecordBatch, IgnitionRuntime, RuntimeError};
use std::any::Any;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::{mem, slice};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, LazyLock};
use ahash::AHasher;
use brotli::enc::static_dict::Hash;
use rustix::mm::{MapFlags, ProtFlags};
use crate::basic::ConvertedType::NONE;

static RUNTIME: LazyLock<Result<IgnitionRuntime, RuntimeError>> = LazyLock::new(|| {
    let mut config = ConfigBuilder::new();
    // config.compile_with_debug(true); // debug mode!
    // config.enable_opentelemetry(true);
    config.compile_with_debug(false);
    config.enable_opentelemetry(false);
    config.set_thread_virtual_memory_limit(5 * 8 * ignition::units::GIB);
    let config = config.into_config();
    let runtime = ignition::build_engine(config);

    runtime
});

/// Ignition version of the PrimitiveArrayReader
pub struct PrimitiveArrayIgnitionReader {
    fd: BorrowedFd<'static>, // not strictly correct, but we know the mapped file outlives Self
    fd_len: usize,
    runtime: &'static IgnitionRuntime,
    job_bundle: Option<HashedIgnitionJob>,
    data_type: ArrowType,
    pages: Box<dyn PageIterator>, // iterates over column chunks (one column chunk per row group)
    cur_page_reader: Option<Box<dyn PageReader>>, // iterates over pages
    column_desc: ColumnDescPtr,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    leftovers: Option<ArrayRef>,
    row_selection: Vec<RowSelector>,
    skip_front_of_next_decoded_page: usize,
    total_rows_read_or_skipped: usize,
    total_num_rows: usize,
}

struct HashedIgnitionJob {
    hash: u64,
    job: IgnitionJob,
    bundle: IgnitionBundle,
}

enum IgnitionPage {
    Decoder {
        buffer: Bytes,
        version: String,
    },
    Data {
        offset: usize,
        length: usize,
        num_rows: u32,
    },
    MemFdData {
        buffer: Bytes,
        num_rows: u32,
        fd: OwnedFd,
    }
}

impl PrimitiveArrayIgnitionReader {
    pub fn new(
        row_groups: &dyn RowGroups,
        mut pages: Box<dyn PageIterator>,
        column_desc: ColumnDescPtr,
        arrow_type: Option<ArrowType>,
    ) -> crate::errors::Result<Self>
    {
        // Check if Arrow type is specified, else create it from Parquet type
        let data_type = match arrow_type {
            Some(t) => t,
            None => parquet_to_arrow_field(column_desc.as_ref())?
                .data_type()
                .clone(),
        };

        // println!("new reader for col: {} with datatype: {}", column_desc.name(), data_type);

        let runtime = match &*RUNTIME {
            Ok(r) => r,
            Err(e) => {
                return Err(ParquetError::External(Box::new(e)));
            }
        };

        // benchmark purposes: clear cache
        // runtime.clear_program_cache();

        let fd = row_groups.file_fd().ok_or_else(|| {
            return general_err!("RowGroups should be a mapped file");
        })??;

        // We know that our file outlives Self
        let fd: BorrowedFd<'static> = unsafe { std::mem::transmute(fd) };

        // in a real system, we need to check that the file is not above the 32bit wasm memory limit
        let fd_len = rustix::fs::fstat(fd).map_err(|e| {
            return general_err!("Unable to get size of fd: {}", fd.as_raw_fd() as usize);
        })?.st_size as usize;

        Ok(Self {
            fd,
            fd_len,
            leftovers: None,
            column_desc,
            runtime,
            job_bundle: None,
            data_type,
            pages,
            cur_page_reader: None,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            row_selection: vec![],
            skip_front_of_next_decoded_page: 0,
            total_rows_read_or_skipped: 0,
            total_num_rows: row_groups.num_rows(),
        })
    }

    fn nullable(&self) -> bool {
        self.column_desc.max_def_level() > 0
    }

    /// gets the next parquet ignition page. Advances row group if needed.
    /// does not apply any skipping logic
    /// returns None if there are no more pages left
    fn get_next_page(&mut self) -> crate::errors::Result<Option<IgnitionPage>> {
        if self.cur_page_reader.is_none() {
            if let Some(page_reader) = self.pages.next() {
                self.cur_page_reader = Some(page_reader?);
            } else {
                return Ok(None);
            }
        }

        let page_reader = self.cur_page_reader.as_mut().unwrap();
        if let Some(page) = page_reader.get_next_page()? {
            match page {
                Page::DataPage { .. } => {
                    return Err(general_err!("DataPage found in Ignition encoded column"));
                }
                Page::DictionaryPage { .. } => {
                    return Err(general_err!(
                        "DictionaryPage found in Ignition encoded column"
                    ));
                }
                Page::DecoderPage { buf, version } => {
                    return Ok(Some(IgnitionPage::Decoder {
                        buffer: buf,
                        version,
                    }));

                }
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed,
                    statistics,
                } => {
                    // here, we handle compressed datav2 pages, which means the pages are in a memfd
                    assert_eq!(encoding, Encoding::IGNITION);
                    assert_eq!(def_levels_byte_len, 0);
                    assert_eq!(rep_levels_byte_len, 0);
                    assert_eq!(num_rows, num_values);
                    assert_eq!(is_compressed, true, "we should only revert to using memfd if the data page is compressed");

                    let fd = ignition::bundle::mapped_fd_from_index(buf.as_ref());

                    let fd = match fd {
                        None => {
                            return Err(general_err!("DataPageV2, but did not find Memfd in Memfd index"));
                        }
                        Some(fd) => {
                            fd
                        }
                    };

                    return Ok(Some(IgnitionPage::MemFdData {
                        buffer: buf,
                        num_rows,
                        fd,
                    }));

                    // return Err(general_err!("DataPageV2 seen, expected MappedDataPageV2"));
                }
                // Page::MappedDecoderPage {
                //     byte_offset,
                //     byte_len,
                //     version,
                // } => {
                //     assert_eq!(version, "1.0.0");
                //     return Ok(Some(IgnitionPage::Decoder {
                //         offset: byte_offset,
                //         length: byte_len,
                //     }));
                // }
                Page::MappedDataPageV2 {
                    byte_offset,
                    byte_len,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed,
                    statistics,
                } => {
                    // store our offset for processing later
                    assert_eq!(encoding, Encoding::IGNITION);
                    assert_eq!(def_levels_byte_len, 0);
                    assert_eq!(rep_levels_byte_len, 0);
                    assert_eq!(num_rows, num_values);
                    assert_eq!(is_compressed, false);

                    return Ok(Some(IgnitionPage::Data {
                        offset: byte_offset,
                        length: byte_len,
                        num_rows,
                    }));
                }
            }
        }

        // finished current page reader
        self.cur_page_reader = None;

        // try again
        self.get_next_page()
    }

    fn handle_new_decoder_page(&mut self, decoder: &[u8], _version: String) -> crate::errors::Result<()> {
        // check if a hash matches for our existing ignition job
        // otherwise, make a new ignition job.
        let mut h = DefaultHasher::new();
        decoder.hash(&mut h);
        let hash = h.finish();

        if let Some(hashed_job) = &self.job_bundle {
            if hashed_job.hash == hash {
                return Ok(());
            }
        }

        // need to create a new bundle

        let schema = Schema::new(vec![Field::new(
            "ignition_col",
            self.data_type.clone(),
            self.nullable(),
        )]);

        let wasm = Vec::from(decoder);
        let bundle = IgnitionBundle::new_extension_borrowed_fd(wasm.clone(), self.fd, self.fd_len, (&schema).into())?;
        let params = ignition::IgnitionJobParametersBuilder::new()
            // .do_not_validate_utf8()
            .finish(&bundle)?;
        // let s = Instant::now();
        let job = self.runtime.init_blocking_job(params)?;
        // println!("init_blocking_job took: {}ms", s.elapsed().as_millis());

        self.job_bundle = Some(HashedIgnitionJob { hash, job, bundle });

        Ok(())
    }

    /// same as decode_batch, but uses native ignition for debug purposes
    fn decode_batch_native(
        &mut self,
        start_tuple: usize,
        num_to_read: usize,
        offset: usize,
        length: usize,
    ) -> crate::errors::Result<ArrayRef> {
        let job = match &mut self.job_bundle {
            Some(job) => job,
            None => {
                return Err(general_err!("Ignition decoder not initialized"));
            }
        };

        job.job.force_offset_range(offset as _, length as _);

        let schema = Schema::new(vec![Field::new(
            "ignition_col",
            self.data_type.clone(),
            self.nullable(),
        )]);
        // let mut native_job = self.runtime.init_native_job("rle_linestatus_paged", (&schema).into())?;
        let mut native_job =
            self.runtime
                .init_native_job("fsst_single_column_paged", (&schema).into(), true)?;
        let fd = match &self.job_bundle.as_ref().unwrap().bundle {
            IgnitionBundle::ExtensionOwned(_, fd, _) => fd.map,
            _ => {
                panic!();
            }
        };

        let data_bytes = unsafe { slice::from_raw_parts(fd.as_ptr().add(offset), length) };
        let ign_record_batch =
            self.runtime
                .run_native_job(&mut native_job, data_bytes, start_tuple, num_to_read)?;

        // below is same as decode_batch
        // dbg!(ign_record_batch.row_count(), ign_record_batch.null_count());
        assert_eq!(ign_record_batch.row_count(), num_to_read);

        let nullable = self.nullable();
        let field = Field::new(self.column_desc.name(), self.data_type.clone(), nullable);
        let schema = Schema::new(vec![std::sync::Arc::new(field)]);

        let record_batch = ign_record_batch.into_arrow_record_batch(std::sync::Arc::new(schema));
        let mut record_batch = record_batch.map_err(|e| RuntimeError::ArrowError(e))?;
        assert_eq!(record_batch.num_columns(), 1);

        // we need to clone out the data from ignition, because it is overwritten on the next invocation
        let decoded = record_batch.remove_column(0);
        let decoded = deep_clone_array(&decoded.into_data())?;
        let mut decoded = arrow_array::make_array(decoded);

        assert_eq!(decoded.len(), num_to_read);

        Ok(decoded)
    }

    /// decodes and clones out the decoded array returned from ignition
    /// if Job is None, uses the job in self.job_bundle
    fn decode_batch_with_job(
        &mut self,
        job: Option<&mut IgnitionJob>,
        start_tuple: usize,
        num_to_read: usize,
    ) -> crate::errors::Result<ArrayRef> {
        // println!("decoding batch for: {}", self.column_desc.name());

        let job = match job {
            None => {
                match &mut self.job_bundle {
                    Some(job) => &mut job.job,
                    None => {
                        return Err(general_err!("Ignition decoder not initialized"));
                    }
                }
            }
            Some(job) => job,
        };

        let ign_record_batch = self
            .runtime
            .run_blocking_job(job, start_tuple, num_to_read)?;
        // dbg!(ign_record_batch.row_count(), ign_record_batch.null_count());
        assert_eq!(ign_record_batch.row_count(), num_to_read);

        let nullable = self.nullable();
        let field = Field::new(self.column_desc.name(), self.data_type.clone(), nullable);
        let schema = Schema::new(vec![std::sync::Arc::new(field)]);

        let record_batch = ign_record_batch.into_arrow_record_batch(std::sync::Arc::new(schema));
        let mut record_batch = record_batch.map_err(|e| RuntimeError::ArrowError(e))?;
        assert_eq!(record_batch.num_columns(), 1);

        // we need to clone out the data from ignition, because it is overwritten on the next invocation
        let decoded = record_batch.remove_column(0);
        let decoded = deep_clone_array(&decoded.into_data())?;
        let mut decoded = arrow_array::make_array(decoded);

        assert_eq!(decoded.len(), num_to_read);

        Ok(decoded)
    }

    /// Handles the decoding of a job and updates the given output array and selection accordingly
    /// if Job is None, uses the job in self.job_bundle
    fn decode_into_array_with_selection(&mut self, array: &mut ArrayRef, selection: &mut RowSelector, num_rows: u32, job: Option<&mut IgnitionJob>) -> Result<(), ParquetError> {
        // decode the incoming data using the current bundle
        assert!(self.skip_front_of_next_decoded_page < num_rows as _);
        let start_tuple = self.skip_front_of_next_decoded_page;
        let num_to_read = num_rows as usize - start_tuple;
        self.skip_front_of_next_decoded_page = 0;

        let decoded = self.decode_batch_with_job(job, 0, num_rows as _)?;

        // dbg!(start_tuple, num_to_read, self.skip_front);
        // let decoded = self.decode_batch_native(start_tuple, num_to_read, offset, length)?;
        // let decoded = self.decode_batch(0, num_rows as _, offset, length)?;

        // this avoids issues with decoders which break with non-zero start_tuple
        let decoded = decoded.slice(start_tuple, num_to_read);

        // copy selection rows into output
        if selection.row_count >= num_to_read {
            // todo: this is very hot, so we can maybe store all the intermediate vecs
            // todo: before, and just concat it once
            *array = arrow_select::concat::concat(&[array.as_ref(), decoded.as_ref()])?;
            selection.row_count -= num_to_read;
        } else {
            assert!(selection.row_count < num_to_read);
            // copy directly into leftovers, handle on next loop iteration
            assert!(self.leftovers.is_none());
            self.leftovers = Some(decoded);

            // selection.row_count remains unchanged
        }
        Ok(())
    }
}

impl ArrayReader for PrimitiveArrayIgnitionReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    #[tracing::instrument(skip(self))]
    fn read_records(&mut self, mut batch_size: usize) -> crate::errors::Result<usize> {
        if batch_size + self.total_rows_read_or_skipped > self.total_num_rows {
            batch_size = self.total_num_rows - self.total_rows_read_or_skipped;
        }
        self.total_rows_read_or_skipped += batch_size;

        if let Some(last) = self.row_selection.last_mut() {
            if last.is_select() {
                last.row_count += batch_size;
            }
            self.row_selection.push(RowSelector::select(batch_size));
        } else {
            self.row_selection.push(RowSelector::select(batch_size));
        }

        Ok(batch_size)
    }

    #[tracing::instrument(skip(self))]
    fn consume_batch(&mut self) -> crate::errors::Result<ArrayRef> {
        // dbg!(self.total_rows_read_or_skipped, self.total_num_rows);
        let mut array = arrow_array::array::new_empty_array(&self.data_type);

        // drain the entire selection
        let mut row_selection = std::mem::replace(&mut self.row_selection, vec![]);
        for mut selection in row_selection.drain(..) {
            // loop until the current selection has been satisfied
            while selection.row_count > 0 {
                // if we reported leftovers, start with them
                if let Some(leftovers) = self.leftovers.take() {
                    if leftovers.len() > selection.row_count {
                        if selection.is_select() {
                            let l = leftovers.slice(0, selection.row_count);
                            array = arrow_select::concat::concat(&[array.as_ref(), l.as_ref()])?;
                        }
                        // in both cases, we store the remainder back into  leftovers
                        let r = leftovers
                            .slice(selection.row_count, leftovers.len() - selection.row_count);
                        self.leftovers = Some(r);
                        selection.row_count = 0;
                    } else if leftovers.len() == selection.row_count {
                        if selection.is_select() {
                            array = arrow_select::concat::concat(&[
                                array.as_ref(),
                                leftovers.as_ref(),
                            ])?;
                        }
                        selection.row_count = 0;
                    } else {
                        assert!(leftovers.len() < selection.row_count);
                        if selection.is_select() {
                            array = arrow_select::concat::concat(&[
                                array.as_ref(),
                                leftovers.as_ref(),
                            ])?;
                        }
                        selection.row_count -= leftovers.len();
                    }
                }

                // we only decode on selects
                // skips that occurred at the end of the previous batch need to be respected
                if selection.is_skip() {
                    self.skip_front_of_next_decoded_page += selection.row_count;
                    selection.row_count = 0;
                    continue;
                }

                // either there is nothing else to do for this RowSelector
                // or we have no more leftovers
                if selection.row_count == 0 {
                    continue;
                }
                assert!(self.leftovers.is_none());

                // decode a new page
                let ignition_page = match self.get_next_page()? {
                    Some(p) => p,
                    None => {
                        dbg!("no pages left");
                        break;
                    }
                };

                match ignition_page {
                    IgnitionPage::Decoder { buffer, version } => {
                        self.handle_new_decoder_page(&buffer, version)?;
                    }
                    IgnitionPage::Data {
                        offset,
                        length,
                        num_rows,
                    } => {
                        // check if we can skip this page
                        if self.skip_front_of_next_decoded_page >= num_rows as _ {
                            // println!("skipped page!");
                            self.skip_front_of_next_decoded_page -= num_rows as usize;
                            continue;
                        }

                        assert!(selection.is_select());

                        // set the job up
                        let mut job = match &mut self.job_bundle {
                            Some(job) => &mut job.job,
                            None => {
                                return Err(general_err!("Ignition decoder not initialized"));
                            }
                        };
                        job.force_offset_range(offset as _, length as _);

                        // {
                        //     // debug
                        //     let fd = match &self.job_bundle.as_ref().unwrap().bundle {
                        //         // IgnitionBundle::ExtensionOwned(_, fd, _) => fd.map,
                        //         IgnitionBundle::ExtensionBorrowed(_, fd, _, _) => fd,
                        //         _ => {
                        //             panic!();
                        //         }
                        //     };
                        //     let fd = fd.try_clone_to_owned().unwrap();
                        //     let mut f = File::from(fd);
                        //     f.seek(SeekFrom::Start(offset as u64)).unwrap();
                        //     let r= f.bytes().take(length).collect::<Vec<_>>();
                        //     let data_bytes = r.iter().map(|b| *b.as_ref().unwrap()).collect::<Vec<_>>();
                        //     let hash = Hash(data_bytes.as_slice());
                        //     println!("{hash}, {length}");
                        // }

                        self.decode_into_array_with_selection(&mut array, &mut selection, num_rows, None)?;
                    }
                    IgnitionPage::MemFdData { buffer, num_rows, fd } => {
                        // check if we can skip this page
                        if self.skip_front_of_next_decoded_page >= num_rows as _ {
                            // println!("skipped page!");
                            self.skip_front_of_next_decoded_page -= num_rows as usize;
                            continue;
                        }

                        assert!(selection.is_select());

                        // {
                        //     // debug
                        //     let hash = Hash(buffer.iter().as_slice());
                        //     println!("{hash}, {}", buffer.len());
                        // }

                        // for this, we want to decode the page using a temporary bundle.
                        let old_bundle = match &self.job_bundle {
                            None => {
                                return Err(general_err!("Job bundle was not seen before MemFdData Data Page"));
                            }
                            Some(bundle) => {
                                &bundle.bundle
                            }
                        };

                        let decoder = old_bundle.decoder();
                        let schema = Schema::new(vec![Field::new(
                            "ignition_col",
                            self.data_type.clone(),
                            self.nullable(),
                        )]);

                        // todo: make new bundle variant to avoid cloning decoder
                        // todo: avoid cloning schema, unify with other schema usage

                        // borrowed_fd is only used in the temp bundle. The temp bundle has a shorter lifetime than the OwnedFd
                        let borrowed_fd: BorrowedFd<'static> = unsafe { mem::transmute(fd.as_fd()) };
                        let temp_bundle = IgnitionBundle::new_extension_borrowed_fd(Vec::from(decoder), borrowed_fd, buffer.len(), (&schema).into())?;

                        let temp_params = ignition::IgnitionJobParametersBuilder::new()
                            .finish(&temp_bundle)?;
                        let mut temp_job = self.runtime.init_blocking_job(temp_params)?;

                        // println!("using tmp job with buffer len: {}", buffer.len());

                        self.decode_into_array_with_selection(&mut array, &mut selection, num_rows, Some(&mut temp_job))?;
                        drop(temp_bundle);
                    }
                }
            }

            assert_eq!(selection.row_count, 0);
        }

        assert!(self.row_selection.is_empty());
        Ok(array)
    }

    fn skip_records(&mut self, mut num_records: usize) -> crate::errors::Result<usize> {
        if num_records + self.total_rows_read_or_skipped > self.total_num_rows {
            num_records = self.total_num_rows - self.total_rows_read_or_skipped;
        }
        self.total_rows_read_or_skipped += num_records;

        if let Some(last) = self.row_selection.last_mut() {
            if last.is_skip() {
                last.row_count += num_records;
            }
            self.row_selection.push(RowSelector::skip(num_records));
        } else {
            self.row_selection.push(RowSelector::skip(num_records));
        }

        Ok(num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_deref()
    }
}

/// We need this in order to copy the data out of the Ignition managed memory.
/// recursively clone
fn deep_clone_array(original: &ArrayData) -> Result<ArrayData, ArrowError> {
    let mut buffers = vec![];
    let mut child_data = vec![];

    for buffer in original.buffers() {
        // let buf = Buffer::from_bytes(Bytes::copy_from_slice(buffer.as_slice()).into());
        // these other options don't work for some reason
        // let buf = Buffer::from(buffer.as_slice());
        let buf = Buffer::from_slice_ref(buffer.as_slice());
        buffers.push(buf);
    }

    for child in original.child_data() {
        child_data.push(deep_clone_array(child)?)
    }

    // normal clone would clone an arc
    let nulls = if let Some(nulls) = original.nulls() {
        let buffer = nulls.buffer();
        let buffer = Buffer::from_slice_ref(buffer.as_slice());
        let bit_len = buffer.len().saturating_mul(8);

        Some(NullBuffer::new(BooleanBuffer::new(buffer, 0, bit_len)))
    } else {
        None
    };

    let array_data = ArrayData::builder(original.data_type().clone())
        .len(original.len())
        .offset(original.offset())
        .buffers(buffers)
        .child_data(child_data)
        .nulls(nulls);

    // Safety: ignition has already validated the utf8 when it is returned.
    // Up to 30% of the samples are spent here validating utf8.
    let array_data = unsafe { array_data.build_unchecked() };

    Ok(array_data)
}
