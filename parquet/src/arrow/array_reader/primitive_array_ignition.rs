use std::any::Any;
use std::cell::OnceCell;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::slice;
use std::sync::LazyLock;
use crate::errors::ParquetError;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringArray};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, UInt8Type};
use arrow_buffer::{Buffer, ToByteSlice};
use arrow_data::ArrayData;
use crate::arrow::array_reader::primitive_array::IntoBuffer;
use crate::arrow::record_reader::RecordReader;
use crate::column::page::{Page, PageIterator, PageReader};
use arrow_schema::{ArrowError, DataType as ArrowType, Field, Schema, TimeUnit};
use bytes::Bytes;
use ignition::bundle::{IgnitionBundle, MappedFd};
use ignition::config::ConfigBuilder;
use ignition::{row, IgnitionJob, IgnitionRuntime, RuntimeError};
use crate::arrow::array_reader::{read_records, ArrayReader, PrimitiveArrayReader, RowGroups};
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::Encoding;
use crate::data_type::ByteArray;
use crate::file::reader::RowGroupReader;
use crate::schema::types::ColumnDescPtr;

static RUNTIME: LazyLock<Result<IgnitionRuntime, RuntimeError>> = LazyLock::new(|| {
    let mut config = ConfigBuilder::new();
    // config.compile_with_debug(true); // debug mode!
    // config.enable_opentelemetry(true);
    // config.validate_utf8(false);
    config.set_thread_virtual_memory_limit(24 * ignition::units::GIB);
    let config = config.into_config();
    let runtime = ignition::build_engine(config);

    runtime
});

/// Ignition version of the PrimitiveArrayReader
pub struct PrimitiveArrayIgnitionReader
where
{
    fd: MappedFd,
    runtime: &'static IgnitionRuntime,
    job_bundle: Option<HashedIgnitionJob>,
    data_type: ArrowType,
    buf: Option<Bytes>,
    pages: Box<dyn PageIterator>, // iterates over column chunks (one column chunk per row group)
    cur_page_reader: Option<Box<dyn PageReader>>, // iterates over pages
    column_desc: ColumnDescPtr,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    leftovers: Option<ArrayRef>,
    reported_len: usize,
    seen_pages: Vec<IgnitionPages>,
}

struct HashedIgnitionJob {
    hash: u64,
    job: IgnitionJob,
    bundle: IgnitionBundle,
}

enum IgnitionPages {
    Decoder {
        offset: usize,
        length: usize,
    },
    Data {
        offset: usize,
        length: usize,
        num_rows: u32,
    }
}


impl PrimitiveArrayIgnitionReader
where
{
    pub fn new(
        row_groups: &dyn RowGroups,
        mut pages: Box<dyn PageIterator>,
        column_desc: ColumnDescPtr,
        arrow_type: Option<ArrowType>,
    ) -> crate::errors::Result<Self> {
        // Check if Arrow type is specified, else create it from Parquet type
        let data_type = match arrow_type {
            Some(t) => t,
            None => parquet_to_arrow_field(column_desc.as_ref())?
                .data_type()
                .clone(),
        };

        // todo: IDK!!
        let runtime = match &*RUNTIME {
            Ok(r) => r,
            Err(e) => panic!("{}", e),
        };

        let fd = row_groups.file_fd().ok_or_else(|| {
            return general_err!("RowGroups should be a mapped file");
        })??;

        Ok(Self {
            fd,
            leftovers: None,
            column_desc,
            runtime,
            job_bundle: None,
            data_type,
            buf: None,
            pages,
            cur_page_reader: None,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            reported_len: 0,
            seen_pages: vec![],
        })
    }
}

impl ArrayReader for PrimitiveArrayIgnitionReader
where
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    // note: seems to be ok to return less than batch_size
    // ParquetRecordBatchReader iterator::next() will call in a loop

    // however, we need to store everything that we read, as consume_batch needs to return everything
    #[tracing::instrument(skip(self))]
    fn read_records(&mut self, batch_size: usize) -> crate::errors::Result<usize> {
        let mut records_read = 0;

        // if we have leftovers, start with that amount
        if let Some(leftovers) = self.leftovers.as_mut() {
            records_read = leftovers.len();
        }

        // otherwise, read until we exceed the batch_size
        assert!(self.seen_pages.is_empty());

        // each row group could have its own decoder
        while records_read < batch_size {
            // if there is no page reader, move to the next row group
            if self.cur_page_reader.is_none() {
                if let Some(page_reader) = self.pages.next() {
                    self.cur_page_reader = Some(page_reader?);
                } else {
                    // we have exhausted all, return
                    break;
                }
            }


            if let Some(page_reader) = self.cur_page_reader.as_mut() {
                if let Some(page) = page_reader.get_next_page()? {
                    match page {
                        Page::DataPage { .. } => {
                            return Err(general_err!("DataPage found in Ignition encoded column"));
                        }
                        Page::DictionaryPage { .. } => {
                            return Err(general_err!("DictionaryPage found in Ignition encoded column"));
                        }
                        Page::DecoderPage { buf, version } => {
                            return Err(general_err!("DecoderPage seen, expected MappedDecoderPage"));
                            // self.seen_pages.push(IgnitionPages::Decoder(buf));
                        }
                        Page::DataPageV2 { buf, num_values, encoding, num_nulls, num_rows, def_levels_byte_len, rep_levels_byte_len, is_compressed, statistics } => {
                            return Err(general_err!("DataPageV2 seen, expected MappedDataPageV2"));
                            // assert_eq!(encoding, Encoding::IGNITION);
                            // assert_eq!(def_levels_byte_len, 0);
                            // assert_eq!(rep_levels_byte_len, 0);
                            // assert_eq!(num_rows, num_values);
                            //
                            // self.seen_pages.push(IgnitionPages::Data(buf, num_values));
                            // records_read += num_values as usize;
                        }
                        Page::MappedDecoderPage { byte_offset, byte_len, version } => {
                            assert_eq!(version, "1.0.0");
                            self.seen_pages.push(IgnitionPages::Decoder {
                                offset: byte_offset,
                                length: byte_len,
                            })
                        }
                        Page::MappedDataPageV2 { byte_offset, byte_len, num_values, encoding, num_nulls, num_rows, def_levels_byte_len, rep_levels_byte_len, is_compressed, statistics } => {
                            // store our offset for processing later
                            assert_eq!(encoding, Encoding::IGNITION);
                            assert_eq!(def_levels_byte_len, 0);
                            assert_eq!(rep_levels_byte_len, 0);
                            assert_eq!(num_rows, num_values);

                            self.seen_pages.push(IgnitionPages::Data {
                                offset: byte_offset,
                                length: byte_len,
                                num_rows,
                            });
                            records_read += num_values as usize;
                        }
                    }
                } else {
                    // finished current page reader
                    self.cur_page_reader = None;
                }
            }
        }

        // say we read up to batch_size rows
        // probs error due to read being called twice in a row
        assert_eq!(self.reported_len, 0);
        self.reported_len = std::cmp::min(records_read, batch_size);
        // dbg!(records_read, batch_size);
        Ok(self.reported_len)
    }

    #[tracing::instrument(skip(self))]
    fn consume_batch(&mut self) -> crate::errors::Result<ArrayRef> {
        let mut total_decoded = 0;

        // if we reported leftovers, start with them
        let mut array = arrow_array::array::new_empty_array(&self.data_type);

        if let Some(leftovers) = self.leftovers.take() {
            if leftovers.len() > self.reported_len {
                let l = leftovers.slice(0, self.reported_len);
                let r = leftovers.slice(self.reported_len, leftovers.len() - self.reported_len);

                self.leftovers = Some(r);
                self.reported_len = 0;
                return Ok(l);
            }
            if leftovers.len() == self.reported_len {
                self.reported_len = 0;
                return Ok(leftovers);
            }

            assert!(leftovers.len() < self.reported_len);
            // array = arrow_select::concat::concat(&[array.as_ref(), leftovers.as_ref()])?;
            total_decoded += leftovers.len() as u32;
            array = leftovers;
        }


        // go through our batches and decode stuff
        for ignition_page in self.seen_pages.drain(..) {
            match ignition_page {
                IgnitionPages::Decoder { offset, length } => {
                    // check if a hash matches for our existing ignition job
                    // otherwise, make a new ignition job.
                    let slice = &self.fd.map[offset..offset + length];
                    let mut h = DefaultHasher::new();
                    slice.hash(&mut h);
                    let hash = h.finish();

                    // if the hashes match, we skip
                    if let Some(hashed_job) = &self.job_bundle {
                        if hashed_job.hash == hash {
                            continue;
                        }
                    }

                    let schema = Schema::new(vec![Field::new("ignition_col", self.data_type.clone(), false)]);

                    // todo: fix up
                    // todo: decoder doesn't need to be mapped
                    if let Some(hashed_job) = self.job_bundle.take() {
                        // replace our job with a new one
                        let fd = match hashed_job.bundle {
                            IgnitionBundle::ExtensionOwned(_, fd, _) => { fd }
                            IgnitionBundle::ExtensionSameFile { fd, .. } => { fd }
                            _ => {
                                return Err(general_err!("Existing ignition bundle was not of correct type"));
                            }
                        };

                        // todo: actually can move to non-owned variant, which saves having to mmap
                        // one file multiple times
                        let wasm = Vec::from(slice);
                        let bundle = IgnitionBundle::new_extension_from_bytes(wasm, Some(fd), (&schema).into())?;

                        let params = ignition::IgnitionJobParametersBuilder::new().finish(&bundle)?;
                        let job = self.runtime.init_blocking_job(params)?;

                        self.job_bundle = Some(HashedIgnitionJob {
                            hash,
                            job,
                            bundle,
                        });
                    } else {
                        // bundle was empty, create a new job
                        let fd = MappedFd::map(&self.fd.fd, self.fd.length())?;
                        // std::mem::swap(&mut fd, &mut self.fd);

                        // this works
                        let wasm = Vec::from(slice);
                        let bundle = IgnitionBundle::new_extension_from_bytes(wasm, Some(fd), (&schema).into())?;


                        let params = ignition::IgnitionJobParametersBuilder::new().finish(&bundle)?;

                        // todo: fails here, let's actually use an ExtensionOwned
                        let job = self.runtime.init_blocking_job(params)?;

                        self.job_bundle = Some(HashedIgnitionJob {
                            hash,
                            job,
                            bundle,
                        });
                    }
                }
                IgnitionPages::Data { offset, length, num_rows } => {
                    // decode the incoming data using the current bundle
                    let job = match &mut self.job_bundle {
                        Some(job) => job,
                        None => {
                            return Err(general_err!("Ignition decoder not initialized"));
                        }
                    };

                    job.job.force_offset_range(offset as _, length as _);

                    // todo, only decode what we need here

                    // dbg!(offset);
                    // run natively for perf debug
                    // let schema = Schema::new(vec![Field::new("ignition_col", self.data_type.clone(), false)]);
                    // // let mut native_job = self.runtime.init_native_job("rle_linestatus_paged", (&schema).into())?;
                    // let mut native_job = self.runtime.init_native_job("fsst_single_column_paged", (&schema).into(), true)?;
                    // let fd = match &job.bundle {
                    //     IgnitionBundle::ExtensionOwned(_, fd, _) => { fd.map }
                    //     _ => {
                    //         panic!();
                    //     }
                    // };

                    // let data_bytes = unsafe { slice::from_raw_parts(fd.as_ptr().add(offset), length) };
                    // let ign_record_batch = self.runtime.run_native_job(&mut native_job, data_bytes, 0, num_rows as usize)?;

                    let ign_record_batch = self.runtime.run_blocking_job(&mut job.job, 0, num_rows as usize)?;
                    // dbg!(ign_record_batch.row_count(), ign_record_batch.null_count());
                    assert_eq!(ign_record_batch.row_count(), num_rows as usize);

                    let nullable = self.column_desc.max_def_level() > 0;
                    let field = Field::new(self.column_desc.name(), self.data_type.clone(), nullable);
                    let schema = Schema::new(vec![std::sync::Arc::new(field)]);

                    let record_batch = ign_record_batch.into_arrow_record_batch(std::sync::Arc::new(schema));
                    let mut record_batch = record_batch.map_err(|e| RuntimeError::ArrowError(e))?;
                    assert_eq!(record_batch.num_columns(), 1);

                    // we need to clone out the data from ignition, because it is overwritten on the next invocation
                    let decoded = record_batch.remove_column(0);
                    let decoded = deep_clone_array(&decoded.into_data())?;
                    let mut decoded = arrow_array::make_array(decoded);

                    assert_eq!(decoded.len(), num_rows as usize);

                    // if we decoded more than we were meant to return, store them as leftovers
                    // we perform the splitting here, as it would avoid an extra concat + split
                    if (num_rows + total_decoded) as usize > self.reported_len {
                        let split_at = self.reported_len - total_decoded as usize;
                        // actually, zero copy slice, may be ok to move out
                        let l = decoded.slice(0, split_at);
                        let r = decoded.slice(split_at, decoded.len() - split_at);
                        decoded = l;


                        assert!(self.leftovers.is_none());
                        self.leftovers = Some(r);
                    }

                    // append to output
                    total_decoded += num_rows;
                    array = arrow_select::concat::concat(&[array.as_ref(), decoded.as_ref()])?;
                }
            }
        }

        // // we actually need to copy out the remainder between ignition invocations
        // // not easy to get arrow to deep copy the buffers
        // // todo: technically, we don't need to clone always, only when buffers are fresh from ignition
        // if let Some(leftovers) = self.leftovers.as_ref() {
        //     let array_data = deep_clone_array(&leftovers.to_data())?;
        //     let cloned = arrow_array::array::make_array(array_data);
        //     self.leftovers = Some(cloned);
        // }

        self.reported_len = 0;
        Ok(array)
    }

    fn skip_records(&mut self, num_records: usize) -> crate::errors::Result<usize> {
        todo!()
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_deref()
    }
}

// recursively clone
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

    let array_data = ArrayData::builder(original.data_type().clone())
        .len(original.len())
        .offset(original.offset())
        .buffers(buffers)
        .child_data(child_data)
        // this clone clones an arc
        .nulls(original.nulls().cloned())
        .build()?;

    Ok(array_data)
}