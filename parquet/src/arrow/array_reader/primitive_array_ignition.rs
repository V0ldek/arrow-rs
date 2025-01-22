use std::any::Any;
use std::hash::{DefaultHasher, Hash, Hasher};
use crate::errors::ParquetError;
use arrow_array::{Array, ArrayRef};
use crate::arrow::array_reader::primitive_array::IntoBuffer;
use crate::arrow::record_reader::RecordReader;
use crate::column::page::{Page, PageIterator, PageReader};
use arrow_schema::{DataType as ArrowType, Field, Schema, TimeUnit};
use bytes::Bytes;
use ignition::bundle::{IgnitionBundle, MappedFd};
use ignition::config::ConfigBuilder;
use ignition::{row, IgnitionJob, IgnitionRuntime, RuntimeError};
use crate::arrow::array_reader::{read_records, ArrayReader, PrimitiveArrayReader, RowGroups};
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::Encoding;
use crate::file::footer::decode_metadata;
use crate::file::reader::RowGroupReader;
use crate::schema::types::ColumnDescPtr;

/// Ignition version of the PrimitiveArrayReader
pub struct PrimitiveArrayIgnitionReader
where
{
    fd: MappedFd,
    runtime: IgnitionRuntime,
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

        let mut config = ConfigBuilder::new();
        config.set_worker_thread_limit(1);
        config.enable_opentelemetry(true);
        //config.compile_with_debug(true); // debug mode!
        let config = config.into_config();
        let runtime = ignition::build_engine(config)?;

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
        self.reported_len = std::cmp::min(records_read, batch_size);
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
                return Ok(l);
            }
            if leftovers.len() == self.reported_len {
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

                        let wasm = Vec::from(slice);
                        let bundle = IgnitionBundle::new_extension_from_bytes(wasm, Some(fd), (&schema).into())?;

                        let params = ignition::IgnitionJobParameters::new(&bundle)?;
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

                        let params = ignition::IgnitionJobParameters::new(&bundle)?;

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

                    // run natively for perf debug
                    // let schema = Schema::new(vec![Field::new("ignition_col", self.data_type.clone(), false)]);
                    // let mut native_job = self.runtime.init_native_job("rle_linestatus_paged", (&schema).into())?;
                    // let fd = match &job.bundle {
                    //     IgnitionBundle::ExtensionOwned(_, fd, _) => { fd.map }
                    //     _ => {
                    //         panic!();
                    //     }
                    // };
                    // unsafe {
                    //     let z = read_metadata(fd.as_ptr().add(offset), num_rows as _);
                    //     dbg!(&z);
                    //     let mut dc = DecodedColumn::new();
                    //     decode_column(&mut dc, &z, 0, num_rows as _);
                    //     // dbg!(String::from_utf8(dc.data.clone()));
                    // }

                    // let ign_record_batch = self.runtime.run_native_job(&mut native_job, fd, 0, num_rows as usize)?;

                    let ign_record_batch = self.runtime.run_blocking_job(&mut job.job, 0, num_rows as usize)?;
                    assert_eq!(ign_record_batch.row_count(), num_rows as usize);

                    let field = Field::new(self.column_desc.name(), self.data_type.clone(), false);
                    let schema = Schema::new(vec![std::sync::Arc::new(field)]);

                    let record_batch = ign_record_batch.into_arrow_record_batch(std::sync::Arc::new(schema));
                    let mut record_batch = record_batch.map_err(|e| RuntimeError::ArrowError(e))?;

                    assert_eq!(record_batch.num_columns(), 1);

                    let mut decoded = record_batch.remove_column(0);
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

        assert_eq!(array.len(), self.reported_len);
        assert_eq!(self.seen_pages.len(), 0);
        Ok(array)
    }

    fn skip_records(&mut self, num_records: usize) -> crate::errors::Result<usize> {
        todo!()
        // we can only skip in page granularity
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_deref()
    }
}

// fsst testing

#[derive(Debug)]
struct FsstColumn {
    table: SymbolTable,
    offsets: *const u32,
    data: *const u8,
}

#[derive(Debug)]
struct SymbolTable {
    symbols: *const u64,
    lens: *const u8,
}
fn read_metadata(data: *const u8, row_count: usize) -> FsstColumn {
    let data_header = data.cast::<u32>();
    read_column_metadata(data, unsafe { data_header.read() }, row_count)
}

fn read_column_metadata(data: *const u8, start: u32, row_count: usize) -> FsstColumn {
    // First is the symbol table.
    let ptr = unsafe { data.add(start as usize) };
    let (symbol_table, ptr_after_symbols) = read_symbol_table_data(ptr);
    // Then we have (row_count + 1) offsets.
    let offsets = ptr_after_symbols.cast::<u32>();
    // And finally the actual data.
    let string_data = unsafe { offsets.add(row_count + 1).cast::<u8>() };

    dbg!(
        "column descr (symboltab, offsets, strings): {:?} {:?} {:?}",
        ptr,
        ptr_after_symbols,
        string_data
    );

    FsstColumn {
        table: symbol_table,
        offsets,
        data: string_data,
    }
}

fn read_symbol_table_data(data: *const u8) -> (SymbolTable, *const u8) {
    // The encoding has 1 + N bytes + padding + 8*N, where N is the number of symbols in the table,
    // and padding is the amount required to have symbols aligned to the 8-byte boundary.
    // |N: u8|len0:u8|len1:u8|...|lenN:u8|opt_padding|sym0: u64|sym1: u64|...|symN:u64|
    let len_byte = unsafe { data.read() };
    let len = len_byte as usize;
    let lens = unsafe { data.add(1) };
    let rem = (len + 1) % 8;
    let symbols_offset = if rem == 0 {
        len + 1
    } else {
        len + 1 + (8 - rem)
    };
    let symbols = unsafe { data.add(symbols_offset).cast::<u64>() };
    let end = unsafe { symbols.add(len).cast::<u8>() };

    let table = SymbolTable { symbols, lens };
    (table, end)
}

#[derive(Debug)]
struct DecodedColumn {
    data: Vec<u8>,
    data_offset: usize,
    validity: Vec<u8>,
    offsets: Vec<u32>,
    null_count: u32,
    validity_byte: u8,
    validity_byte_idx: usize,
}


impl DecodedColumn {
    fn new() -> Self {
        DecodedColumn {
            data: Vec::new(),
            data_offset: 0,
            validity: Vec::new(),
            offsets: Vec::new(),
            null_count: 0,
            validity_byte: 0,
            validity_byte_idx: 0,
        }
    }

    fn prepare(&mut self, tuple_count: usize, total_compressed_len: usize) {
        self.data.clear();
        self.data.reserve(2 * total_compressed_len);
        self.data_offset = 0;
        self.validity.clear();
        self.validity.reserve((tuple_count + 7) / 8);
        self.offsets.clear();
        self.offsets.reserve(tuple_count);
        self.null_count = 0;
        self.validity_byte = 0;
        self.validity_byte_idx = 0;
        self.offsets.push(0);
    }

    fn ptr_for_next_value(&mut self, max_len: usize) -> *mut u8 {
        let start = self.data_offset;
        let end = self.data_offset + max_len;
        if end >= self.data.len() {
            self.data.resize(end + 8, 0);
        }
        unsafe { self.data.as_mut_ptr().add(start) }
    }

    fn commit_value(&mut self, len: usize) {
        self.validity_byte |= 1 << self.validity_byte_idx;
        self.validity_byte_idx += 1;

        self.try_push_validity();

        self.data_offset += len;
        self.offsets.push(self.data_offset as u32);
    }

    fn push_null(&mut self) {
        self.validity_byte_idx += 1;
        self.null_count += 1;

        self.try_push_validity();
        self.offsets.push(self.data_offset as u32);
    }

    fn try_push_validity(&mut self) {
        if self.validity_byte_idx == 8 {
            self.force_push_validity();
        }
    }

    fn force_push_validity(&mut self) {
        self.validity.push(self.validity_byte);
        self.validity_byte = 0;
        self.validity_byte_idx = 0;
    }

    fn finish(&mut self) {
        if self.validity_byte_idx != 0 {
            self.force_push_validity();
        }
    }

    fn null_count(&self) -> u32 {
        self.null_count
    }

    fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn validity_ptr(&self) -> *const u8 {
        self.validity.as_ptr()
    }

    fn offsets_ptr(&self) -> *const u8 {
        self.offsets.as_ptr().cast()
    }
}

const FSST_ESCAPE: u8 = 0xFF;

unsafe fn decode_column(
    column: &mut DecodedColumn,
    compressed: &FsstColumn,
    start_tuple: usize,
    tuple_count: usize,
) {
    dbg!(
        "decode column data, offsets: {:?} {:?}",
        compressed.data,
        compressed.offsets
    );
    let offset_of_first = compressed.offsets.add(start_tuple).read() as usize;
    let offset_of_after_last = compressed.offsets.add(start_tuple + tuple_count).read() as usize;
    // dbg!("offsets: {} to {}", offset_of_first, offset_of_after_last);
    column.prepare(tuple_count, offset_of_after_last - offset_of_first);

    for tuple_idx in start_tuple..(start_tuple + tuple_count) {
        // dbg!("tuple_idx: {}", tuple_idx);
        let start_offset = compressed.offsets.add(tuple_idx).read() as usize;
        let end_offset = compressed.offsets.add(tuple_idx + 1).read() as usize;
        let compressed_len = end_offset - start_offset;
        // dbg!("offsets: {} to {}", start_offset, end_offset);

        if compressed_len == 0 {
            column.push_null();
        } else {
            // dbg!("requesting ptr_for_next_value: {}", compressed_len * 8);
            let ptr = column.ptr_for_next_value(compressed_len * 8);
            let mut read_i = 0;
            let mut write_i = 0;
            while read_i < compressed_len {
                let b = compressed.data.add(start_offset + read_i).read();
                read_i += 1;

                if b == FSST_ESCAPE {
                    let b = compressed.data.add(start_offset + read_i).read();
                    // dbg!("WRITE 8: {:?}", ptr.add(write_i));
                    ptr.add(write_i).write(b);
                    read_i += 1;
                    write_i += 1;
                } else {
                    let len = compressed.table.lens.add(b as usize).read();
                    let symbol = compressed.table.symbols.add(b as usize).read();
                    // dbg!("WRITE 64: {:?}", ptr.add(write_i));
                    ptr.add(write_i).cast::<u64>().write_unaligned(symbol);
                    write_i += len as usize;
                }
            }
            column.commit_value(write_i);
        }
    }

    column.finish();
}