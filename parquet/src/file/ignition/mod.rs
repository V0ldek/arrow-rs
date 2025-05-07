use crate::compression::{self, Codec, CodecOptionsBuilder};
use crate::format as parquet;
use crate::format::{ColumnIndex, DataPageHeaderV2, OffsetIndex};
use crate::thrift::TSerializable;
use crate::{basic::Encoding, bloom_filter::Sbbf};
use crate::{
    basic::{Compression, ConvertedType, LogicalType, PageType, Type},
    data_type::private::ParquetValueType,
};
use bytes::Bytes;
use std::fmt::Debug;
use std::io::{BufWriter, IoSlice, Read};
use std::{io::Write, sync::Arc};
use std::collections::BTreeSet;
use thrift::protocol::TCompactOutputProtocol;

use crate::column::writer::{get_typed_column_writer_mut, ColumnCloseResult, ColumnMetrics, ColumnWriterImpl, PageMetrics};
use crate::column::{
    page::{CompressedPage, Page, PageWriteSpec, PageWriter},
    writer::{get_column_writer, ColumnWriter},
};
use crate::data_type::{AsBytes, DataType};
use crate::encodings::levels::LevelEncoder;
use crate::errors::{ParquetError, Result};
use crate::file::properties::{BloomFilterPosition, EnabledStatistics, WriterPropertiesPtr};
use crate::file::reader::ChunkReader;
use crate::file::writer::{OnCloseColumnChunk, TrackedWrite};
use crate::file::{
    metadata::ColumnChunkMetaData,
    properties::{WriterProperties, WriterVersion},
    statistics::{Statistics, ValueStatistics},
};
use crate::file::{metadata::*, PARQUET_MAGIC};
use crate::schema::types::{
    ColumnDescPtr, ColumnDescriptor, SchemaDescPtr, SchemaDescriptor, TypePtr,
};

pub struct IgnitionColumnWriter<'a, T: Default, W: Write> {
    sink: &'a mut TrackedWrite<W>,
    on_close: OnCloseColumnChunk<'a>,
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
    statistics_enabled: EnabledStatistics,
    codec: Compression,
    compressor: Option<Box<dyn Codec>>,

    column_metrics: ColumnMetrics<T>,

    column_index_builder: ColumnIndexBuilder,
    offset_index_builder: OffsetIndexBuilder,
}

impl<'a, T: ParquetValueType, W: Write> IgnitionColumnWriter<'a, T, W> {
    pub(crate) fn new(
        column: ColumnDescPtr,
        buf: &'a mut TrackedWrite<W>,
        props: WriterPropertiesPtr,
        on_close: OnCloseColumnChunk<'a>,
    ) -> Result<Self> {
        let codec = props.compression(column.path());
        let codec_options = CodecOptionsBuilder::default().build();
        let compressor = compression::create_codec(codec, &codec_options).unwrap();

        let mut encodings = BTreeSet::new();
        // Used for level information
        encodings.insert(Encoding::RLE);

        let statistics_enabled = props.statistics_enabled(column.path());
        let mut column_metrics = ColumnMetrics::new();
        // let mut page_metrics = PageMetrics::new();

        // Initialize level histograms if collecting page or chunk statistics
        if statistics_enabled != EnabledStatistics::None {
            column_metrics = column_metrics
                .with_repetition_level_histogram(column.max_rep_level())
                .with_definition_level_histogram(column.max_def_level())
        }

        // Disable column_index_builder if not collecting page statistics.
        let mut column_index_builder = ColumnIndexBuilder::new();
        if statistics_enabled != EnabledStatistics::Page {
            column_index_builder.to_invalid()
        }

        Ok(Self {
            descr: column,
            sink: buf,
            on_close,
            props,
            codec,
            compressor,
            statistics_enabled,
            column_metrics,
            column_index_builder,
            offset_index_builder: OffsetIndexBuilder::new(),
        })
    }

    pub fn write_decoder(
        &mut self,
        decoder_bytes: &[u8],
        version: impl Into<String>,
    ) -> Result<()> {
        dbg!("write decoder");
        let uncompressed_size = decoder_bytes.len();

        let buf = if let Some(ref mut cmpr) = self.compressor {
            let mut compressed_buf = Vec::with_capacity(uncompressed_size);
            cmpr.compress(decoder_bytes, &mut compressed_buf)?;
            println!("writing compressed decoder. Original size: {}, compressed size: {}", uncompressed_size, compressed_buf.len());
            compressed_buf.into()
        } else {
            Bytes::copy_from_slice(decoder_bytes)
        };

        let page = Page::DecoderPage {
            buf,
            version: version.into(),
        };

        let compressed_page = CompressedPage::new(page, uncompressed_size);

        let page_spec = self.write_page(compressed_page)?;
        self.update_metrics_for_page(page_spec);
        // For the directory page, don't need to update column/offset index.
        Ok(())
    }

    /// uncompressed_len here refers to the size of page_data
    pub fn write_data_page(
        &mut self,
        page_data: &[u8],
        mut min: Option<T>,
        mut max: Option<T>,
        num_nulls: u32,
        num_values: u32,
        num_distinct: Option<u64>,
        uncompressed_len: usize,
        parquet_double_compress: bool,
        page_variable_length_bytes: Option<i64>, // only set for byte array. total length.
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) -> Result<()> {
        // dbg!("write data", num_values);
        assert_eq!(page_data.len(), uncompressed_len);

        // this uses parquet compression ON TOP of the existing ignition buffer
        let buf = if parquet_double_compress {
            if let Some(ref mut cmpr) = self.compressor {
                let mut compressed_buf = Vec::with_capacity(uncompressed_len);
                cmpr.compress(page_data, &mut compressed_buf)?;
                compressed_buf.into()
            } else {
                panic!("parquet_double_compress was true, but column does not have compressor");
            }
        } else {
            Bytes::copy_from_slice(page_data)
        };

        {
            // done elsewhere (disable when writing nested?)
            self.column_metrics.total_rows_written += u64::from(num_values);
            self.column_metrics.num_column_nulls += u64::from(num_nulls);
        }

        if let Some(page_min) = &min {
            update_min(
                &self.descr,
                page_min,
                &mut self.column_metrics.min_column_value,
            );
        }
        if let Some(page_max) = &max {
            update_max(
                &self.descr,
                page_max,
                &mut self.column_metrics.max_column_value,
            );
        }

        let statistics = if self.can_truncate_value() {
            let mut did_truncate_min = false;
            let mut did_truncate_max = false;
            let mut new_min = None;
            let mut new_max = None;

            if min.is_some() {
                let (trunc_min, did_truncate_min_new) = truncate_min_value(
                    self.props.statistics_truncate_length(),
                    min.as_ref().map(AsBytes::as_bytes).unwrap(),
                );

                did_truncate_min = did_truncate_min_new;
                new_min = Some(trunc_min.into());
            }

            if max.is_some() {
                let (trunc_max, did_truncate_max_new) = truncate_max_value(
                    self.props.statistics_truncate_length(),
                    max.as_ref().map(AsBytes::as_bytes).unwrap(),
                );

                did_truncate_max = did_truncate_max_new;
                new_max = Some(trunc_max.into());
            }

            Statistics::ByteArray(
                ValueStatistics::new(
                    new_min,
                    new_max,
                    num_distinct,
                    Some(num_nulls.into()),
                    false,
                )
                    .with_min_is_exact(!did_truncate_max)
                    .with_max_is_exact(!did_truncate_max),
            )
        } else {
            let vs = ValueStatistics::new(
                min.clone(),
                max.clone(),
                num_distinct,
                Some(num_nulls.into()),
                false,
            );
            Statistics::from(vs)
        };


        let page = Page::DataPageV2 {
            buf,
            num_values,
            encoding: crate::basic::Encoding::IGNITION,
            num_nulls,
            // num_rows: num_values + num_nulls,
            num_rows: num_values,
            def_levels_byte_len: 0,
            rep_levels_byte_len: 0,
            is_compressed: parquet_double_compress, // this refers to parquet's builtin compression
            statistics: Some(statistics),
        };
        let compressed_page = CompressedPage::new(page, uncompressed_len);

        // this only works for vortex pages
        let page_spec = self.write_page_aligned(compressed_page, def_levels, rep_levels)?;
        // let page_spec = self.write_page(compressed_page)?;

        // update column index
        let null_page = num_values == num_nulls;

        if null_page && self.column_index_builder.valid() {
            self.column_index_builder.append(
                null_page,
                vec![],
                vec![],
                num_nulls as _,
            );
        } else if self.column_index_builder.valid() {
            if !(min.is_some() && max.is_some()) {
                self.column_index_builder.to_invalid();
            } else {
                self.column_index_builder.append(
                    null_page,
                    min.as_ref().map(|m| AsBytes::as_bytes(m)).unwrap().to_vec(),
                    max.as_ref().map(|m| AsBytes::as_bytes(m)).unwrap().to_vec(),
                    num_nulls as _,
                );
            }
        }

        // handle rep/def level
        if let (Some(def), Some(rep)) = (def_levels, rep_levels) {
            if def.len() != rep.len() {
                return Err(general_err!(
                    "Inconsistent length of definition and repetition levels: {} != {}",
                    def.len(),
                    rep.len()
                ));
            }
        }
        let mut repetition_level_histogram = LevelHistogram::try_new(self.descr.max_rep_level());
        let mut definition_level_histogram = LevelHistogram::try_new(self.descr.max_def_level());

        // if self.descr.max_def_level() > 0 {
        //     let levels = def_levels.ok_or_else(|| {
        //         general_err!(
        //             "Definition levels are required, because max definition level = {}",
        //             self.descr.max_def_level()
        //         )
        //     })?;
        //
        //     if let Some(ref mut def_hist) = definition_level_histogram {
        //         def_hist.update_from_levels(levels);
        //     }
        // }

        if self.descr.max_rep_level() > 0 {
            // A row could contain more than one value.
            let levels = rep_levels.ok_or_else(|| {
                general_err!(
                    "Repetition levels are required, because max repetition level = {}",
                    self.descr.max_rep_level()
                )
            })?;

            if !levels.is_empty() && levels[0] != 0 {
                return Err(general_err!(
                    "Write must start at a record boundary, got non-zero repetition level of {}",
                    levels[0]
                ));
            }

            if let Some(ref mut def_hist) = repetition_level_histogram {
                def_hist.update_from_levels(levels);
            }
        }

        ColumnMetrics::<T>::update_histogram(&mut self.column_metrics.repetition_level_histogram, &repetition_level_histogram);
        ColumnMetrics::<T>::update_histogram(&mut self.column_metrics.definition_level_histogram, &definition_level_histogram);
        self.column_index_builder.append_histograms(
            &repetition_level_histogram,
            &definition_level_histogram,
        );

        // Update the offset index
        self.offset_index_builder
            .append_row_count(num_values as _);

        self.offset_index_builder
            .append_unencoded_byte_array_data_bytes(page_variable_length_bytes);

        self.offset_index_builder
            .append_offset_and_size(page_spec.offset as i64, page_spec.compressed_size as i32);

        self.update_metrics_for_page(page_spec);
        Ok(())
    }

    fn write_page(&mut self, page: CompressedPage) -> Result<PageWriteSpec> {
        let page_type = page.page_type();
        let start_pos = self.sink.bytes_written() as u64;

        let page_header = page.to_thrift_header();
        let header_size = self.serialize_page_header(page_header)?;
        self.sink.write_all(page.data())?;

        let mut spec = PageWriteSpec::new();
        spec.page_type = page_type;
        spec.uncompressed_size = page.uncompressed_size() + header_size;
        spec.compressed_size = page.compressed_size() + header_size;
        spec.offset = start_pos;
        spec.bytes_written = self.sink.bytes_written() as u64 - start_pos;
        spec.num_values = page.num_values();

        Ok(spec)
    }

    /// THIS IS FOR THE "VORTEX" PAGED FORMAT with the following format in the page.
    /// doesn't quite work, because our tracked write is only for this column.
    /// this assumes we are writing a page with the following format.
    /// first, rep/def buffers.
    ///
    /// first 4 bytes: tuples/chunk u32
    /// next 4 bytes: start offset of data chunk u32
    /// next 4 bytes: end offset of data chunk u32
    /// some padding bytes in order to 8-byte align the data chunk in the file
    /// data chunk
    /// what this function will do is make sure start offset is aligned to 4 bytes
    fn write_page_aligned(&mut self, page: CompressedPage, def_levels: Option<&[i16]>, rep_levels: Option<&[i16]>) -> Result<PageWriteSpec> {
        fn encode_levels_v2(levels: &[i16], max_level: i16) -> Vec<u8> {
            let mut encoder = LevelEncoder::v2(max_level, levels.len());
            encoder.put(levels);
            encoder.consume()
        }

        let encoded_def = def_levels.map(|lv| encode_levels_v2(lv, self.descr.max_def_level()));
        let encoded_rep = rep_levels.map(|lv| encode_levels_v2(lv, self.descr.max_rep_level()));
        let def_levels_byte_len = encoded_def.as_ref().map(|b| b.len()).unwrap_or(0);
        let rep_levels_byte_len = encoded_rep.as_ref().map(|b| b.len()).unwrap_or(0);
        // let uncompressed_size = rep_levels_byte_len + def_levels_byte_len + values_data.buf.len();

        let page_type = page.page_type();
        let start_pos = self.sink.bytes_written() as u64;

        let mut page_header = page.to_thrift_header();

        // handle the def/rep stuff, plae it at the front.
        if let DataPageHeaderV2 { definition_levels_byte_length, repetition_levels_byte_length, .. } = page_header.data_page_header_v2.as_mut().unwrap() {
            *definition_levels_byte_length = def_levels_byte_len as _;
            *repetition_levels_byte_length = rep_levels_byte_len as _;
        }
        page_header.uncompressed_page_size += def_levels_byte_len as i32;
        page_header.uncompressed_page_size += rep_levels_byte_len as i32;

        if let Some(def) = encoded_def {
            self.sink.write(def.as_slice())?;
        }
        if let Some(rep) = encoded_rep {
            self.sink.write(rep.as_slice())?;
        }


        // pre-write the header to a buffer because we need the header's size
        let predicted_header_size = {
            let mut fake_sink = vec![];
            let mut tracked_sink = TrackedWrite::new(&mut fake_sink);
            let mut protocol = TCompactOutputProtocol::new(&mut tracked_sink);
            page_header.write_to_out_protocol(&mut protocol)?;
            let header_size = tracked_sink.bytes_written();
            header_size
        };
        let predicted_padding = {
            let mut cur_pos = self.sink.bytes_written() as u64;
            cur_pos += predicted_header_size as u64;
            cur_pos += 4;

            let padding = (8 - (cur_pos % 8)) % 8;
            padding
        };

        page_header.compressed_page_size += predicted_padding as i32;
        // page_header.uncompressed_page_size += predicted_padding as i32;

        let header_size = self.serialize_page_header(page_header.clone())?;

        dbg!(predicted_header_size, header_size);
        assert_eq!(predicted_header_size, header_size);

        // write the tuples-per-page value (4 bytes)
        self.sink.write_all(&page.data()[0..4])?;

        self.sink.flush()?;

        // now check to see how we are looking in terms of alignment
        let mut cur_pos = self.sink.bytes_written() as u64;
        // cur_pos += std::fs::read("/home/maurice/IdeaProjects/portable-decompress/tmp/tpch/duckdb_snappy/nation.parquet").unwrap().len() as u64;
        // if we are 8 byte aligned here we are good
        let padding = (8 - (cur_pos % 8)) % 8;
        // dbg!(cur_pos, padding);

        // write start
        let start_offset = (12 + padding) as u32;
        self.sink.write_all(&start_offset.to_le_bytes())?;
        // write end
        let original_end = u32::from_le_bytes(page.data()[8..12].try_into().unwrap());
        let end = original_end + padding as u32;
        self.sink.write_all(&end.to_le_bytes())?;

        // dbg!(start_offset, end);

        // now we write the padding
        let padding_bytes = std::iter::repeat(0).take(padding as usize).collect::<Vec<u8>>();
        self.sink.write_all(padding_bytes.as_slice())?;

        // now write the data (should be aligned)
        // assert_eq!((8 - (self.sink.bytes_written() % 8)) % 8, 0);
        self.sink.write_all(&page.data()[12..])?;

        // handle rest
        let mut spec = PageWriteSpec::new();
        spec.page_type = page_type;
        spec.uncompressed_size = page.uncompressed_size() + header_size + padding as usize;
        spec.compressed_size = page.compressed_size() + header_size + padding as usize;
        spec.offset = start_pos;
        spec.bytes_written = self.sink.bytes_written() as u64 - start_pos;
        spec.num_values = page.num_values();

        dbg!(&spec.num_values);

        Ok(spec)
    }

    #[inline]
    fn serialize_page_header(&mut self, header: parquet::PageHeader) -> Result<usize> {
        let start_pos = self.sink.bytes_written();
        {
            let mut protocol = TCompactOutputProtocol::new(&mut self.sink);
            header.write_to_out_protocol(&mut protocol)?;
        }
        Ok(self.sink.bytes_written() - start_pos)
    }

    /// Updates column writer metrics with each page metadata.
    #[inline]
    fn update_metrics_for_page(&mut self, page_spec: PageWriteSpec) {
        self.column_metrics.total_uncompressed_size += page_spec.uncompressed_size as u64;
        self.column_metrics.total_compressed_size += page_spec.compressed_size as u64;
        self.column_metrics.total_bytes_written += page_spec.bytes_written;

        match page_spec.page_type {
            PageType::DATA_PAGE | PageType::DATA_PAGE_V2 => {
                self.column_metrics.total_num_values += page_spec.num_values as u64;
                if self.column_metrics.data_page_offset.is_none() {
                    self.column_metrics.data_page_offset = Some(page_spec.offset);
                }
            }
            PageType::DICTIONARY_PAGE | PageType::DECODER_PAGE => {
                assert!(
                    self.column_metrics.dictionary_page_offset.is_none(),
                    "Dictionary offset is already set"
                );
                // re-use the dictionary_page_offset for the ignition decoder offset
                self.column_metrics.dictionary_page_offset = Some(page_spec.offset);
            }
            _ => {}
        }
    }

    pub fn get_empty_bloom_filter(&self) -> Result<Option<Sbbf>> {
        self.props
            .bloom_filter_properties(self.descr.path())
            .map(|props| Sbbf::new_with_ndv_fpp(props.ndv, props.fpp))
            .transpose()
    }

    pub fn close(mut self, bloom_filter: Option<Sbbf>) -> Result<()> {
        let metadata = self.build_column_metadata()?;
        self.sink.flush()?;

        let column_index = self
            .column_index_builder
            .valid()
            .then(|| self.column_index_builder.build_to_thrift());
        let offset_index = Some(self.offset_index_builder.build_to_thrift());

        let result = ColumnCloseResult {
            bytes_written: self.column_metrics.total_bytes_written,
            rows_written: self.column_metrics.total_rows_written,
            bloom_filter,
            metadata,
            column_index,
            offset_index,
        };

        (self.on_close)(result)?;

        Ok(())
    }

    /// Assembles column chunk metadata.
    fn build_column_metadata(&mut self) -> Result<ColumnChunkMetaData> {
        let total_compressed_size = self.column_metrics.total_compressed_size as i64;
        let total_uncompressed_size = self.column_metrics.total_uncompressed_size as i64;
        let num_values = self.column_metrics.total_num_values as i64;
        let dict_page_offset = self.column_metrics.dictionary_page_offset.map(|v| v as i64);
        // If data page offset is not set, then no pages have been written
        let data_page_offset = self.column_metrics.data_page_offset.unwrap_or(0) as i64;

        let mut builder = ColumnChunkMetaData::builder(self.descr.clone())
            .set_compression(self.codec)
            .set_encodings(vec![Encoding::IGNITION])
            .set_total_compressed_size(total_compressed_size)
            .set_total_uncompressed_size(total_uncompressed_size)
            .set_num_values(num_values)
            .set_data_page_offset(data_page_offset)
            .set_dictionary_page_offset(dict_page_offset);

        if self.statistics_enabled != EnabledStatistics::None {
            let backwards_compatible_min_max = self.descr.sort_order().is_signed();

            let statistics = ValueStatistics::<T>::new(
                self.column_metrics.min_column_value.clone(),
                self.column_metrics.max_column_value.clone(),
                self.column_metrics.column_distinct_count,
                Some(self.column_metrics.num_column_nulls),
                false,
            )
            .with_backwards_compatible_min_max(backwards_compatible_min_max)
            .into();

            let statistics = match statistics {
                Statistics::ByteArray(stats) if stats._internal_has_min_max_set() => {
                    let (min, did_truncate_min) = truncate_min_value(
                        self.props.statistics_truncate_length(),
                        stats.min_bytes_opt().unwrap(),
                    );
                    let (max, did_truncate_max) = truncate_max_value(
                        self.props.statistics_truncate_length(),
                        stats.max_bytes_opt().unwrap(),
                    );
                    Statistics::ByteArray(
                        ValueStatistics::new(
                            Some(min.into()),
                            Some(max.into()),
                            stats.distinct_count(),
                            stats.null_count_opt(),
                            backwards_compatible_min_max,
                        )
                        .with_max_is_exact(!did_truncate_max)
                        .with_min_is_exact(!did_truncate_min),
                    )
                }
                Statistics::FixedLenByteArray(stats)
                    if (stats._internal_has_min_max_set() && self.can_truncate_value()) =>
                {
                    let (min, did_truncate_min) = truncate_min_value(
                        self.props.statistics_truncate_length(),
                        stats.min_bytes_opt().unwrap(),
                    );
                    let (max, did_truncate_max) = truncate_max_value(
                        self.props.statistics_truncate_length(),
                        stats.max_bytes_opt().unwrap(),
                    );
                    Statistics::FixedLenByteArray(
                        ValueStatistics::new(
                            Some(min.into()),
                            Some(max.into()),
                            stats.distinct_count(),
                            stats.null_count_opt(),
                            backwards_compatible_min_max,
                        )
                        .with_max_is_exact(!did_truncate_max)
                        .with_min_is_exact(!did_truncate_min),
                    )
                }
                stats => stats,
            };

            builder = builder
                .set_statistics(statistics)
                .set_unencoded_byte_array_data_bytes(self.column_metrics.variable_length_bytes)
                .set_repetition_level_histogram(
                    self.column_metrics.repetition_level_histogram.take(),
                )
                .set_definition_level_histogram(
                    self.column_metrics.definition_level_histogram.take(),
                );
        }

        let metadata = builder.build()?;
        Ok(metadata)
    }

    /// Determine if we should allow truncating min/max values for this column's statistics
    fn can_truncate_value(&self) -> bool {
        match self.descr.physical_type() {
            // Don't truncate for Float16 and Decimal because their sort order is different
            // from that of FIXED_LEN_BYTE_ARRAY sort order.
            // So truncation of those types could lead to inaccurate min/max statistics
            Type::FIXED_LEN_BYTE_ARRAY
                if !matches!(
                    self.descr.logical_type(),
                    Some(LogicalType::Decimal { .. }) | Some(LogicalType::Float16)
                ) =>
            {
                true
            }
            Type::BYTE_ARRAY => true,
            // Truncation only applies for fba/binary physical types
            _ => false,
        }
    }
}

fn update_min<T: ParquetValueType>(descr: &ColumnDescriptor, val: &T, min: &mut Option<T>) {
    update_stat::<T, _>(descr, val, min, |cur| compare_greater(descr, cur, val))
}

fn update_max<T: ParquetValueType>(descr: &ColumnDescriptor, val: &T, max: &mut Option<T>) {
    update_stat::<T, _>(descr, val, max, |cur| compare_greater(descr, val, cur))
}

/// Perform a conditional update of `cur`, skipping any NaN values
///
/// If `cur` is `None`, sets `cur` to `Some(val)`, otherwise calls `should_update` with
/// the value of `cur`, and updates `cur` to `Some(val)` if it returns `true`
fn update_stat<T: ParquetValueType, F>(
    descr: &ColumnDescriptor,
    val: &T,
    cur: &mut Option<T>,
    should_update: F,
) where
    F: Fn(&T) -> bool,
{
    if is_nan(descr, val) {
        return;
    }

    if cur.as_ref().map_or(true, should_update) {
        *cur = Some(val.clone());
    }
}

#[inline]
#[allow(clippy::eq_op)]
fn is_nan<T: ParquetValueType>(descr: &ColumnDescriptor, val: &T) -> bool {
    match T::PHYSICAL_TYPE {
        Type::FLOAT | Type::DOUBLE => val != val,
        Type::FIXED_LEN_BYTE_ARRAY if descr.logical_type() == Some(LogicalType::Float16) => {
            unimplemented!()
        }
        _ => false,
    }
}

/// Evaluate `a > b` according to underlying logical type.
fn compare_greater<T: ParquetValueType>(descr: &ColumnDescriptor, a: &T, b: &T) -> bool {
    if let Some(LogicalType::Integer { is_signed, .. }) = descr.logical_type() {
        if !is_signed {
            // need to compare unsigned
            return a.as_u64().unwrap() > b.as_u64().unwrap();
        }
    }

    match descr.converted_type() {
        ConvertedType::UINT_8
        | ConvertedType::UINT_16
        | ConvertedType::UINT_32
        | ConvertedType::UINT_64 => {
            return a.as_u64().unwrap() > b.as_u64().unwrap();
        }
        _ => {}
    };

    if let Some(LogicalType::Decimal { .. }) = descr.logical_type() {
        match T::PHYSICAL_TYPE {
            Type::FIXED_LEN_BYTE_ARRAY | Type::BYTE_ARRAY => {
                return compare_greater_byte_array_decimals(a.as_bytes(), b.as_bytes());
            }
            _ => {}
        };
    }

    if descr.converted_type() == ConvertedType::DECIMAL {
        match T::PHYSICAL_TYPE {
            Type::FIXED_LEN_BYTE_ARRAY | Type::BYTE_ARRAY => {
                return compare_greater_byte_array_decimals(a.as_bytes(), b.as_bytes());
            }
            _ => {}
        };
    };

    if let Some(LogicalType::Float16) = descr.logical_type() {
        unimplemented!()
    }

    a > b
}

/// Signed comparison of bytes arrays
fn compare_greater_byte_array_decimals(a: &[u8], b: &[u8]) -> bool {
    let a_length = a.len();
    let b_length = b.len();

    if a_length == 0 || b_length == 0 {
        return a_length > 0;
    }

    let first_a: u8 = a[0];
    let first_b: u8 = b[0];

    // We can short circuit for different signed numbers or
    // for equal length bytes arrays that have different first bytes.
    // The equality requirement is necessary for sign extension cases.
    // 0xFF10 should be equal to 0x10 (due to big endian sign extension).
    if (0x80 & first_a) != (0x80 & first_b) || (a_length == b_length && first_a != first_b) {
        return (first_a as i8) > (first_b as i8);
    }

    // When the lengths are unequal and the numbers are of the same
    // sign we need to do comparison by sign extending the shorter
    // value first, and once we get to equal sized arrays, lexicographical
    // unsigned comparison of everything but the first byte is sufficient.

    let extension: u8 = if (first_a as i8) < 0 { 0xFF } else { 0 };

    if a_length != b_length {
        let not_equal = if a_length > b_length {
            let lead_length = a_length - b_length;
            a[0..lead_length].iter().any(|&x| x != extension)
        } else {
            let lead_length = b_length - a_length;
            b[0..lead_length].iter().any(|&x| x != extension)
        };

        if not_equal {
            let negative_values: bool = (first_a as i8) < 0;
            let a_longer: bool = a_length > b_length;
            return if negative_values { !a_longer } else { a_longer };
        }
    }

    (a[1..]) > (b[1..])
}

fn truncate_min_value(truncation_length: Option<usize>, data: &[u8]) -> (Vec<u8>, bool) {
    truncation_length
        .filter(|l| data.len() > *l)
        .and_then(|l| match std::str::from_utf8(data) {
            Ok(str_data) => truncate_utf8(str_data, l),
            Err(_) => Some(data[..l].to_vec()),
        })
        .map(|truncated| (truncated, true))
        .unwrap_or_else(|| (data.to_vec(), false))
}

fn truncate_max_value(truncation_length: Option<usize>, data: &[u8]) -> (Vec<u8>, bool) {
    truncation_length
        .filter(|l| data.len() > *l)
        .and_then(|l| match std::str::from_utf8(data) {
            Ok(str_data) => truncate_utf8(str_data, l).and_then(increment_utf8),
            Err(_) => increment(data[..l].to_vec()),
        })
        .map(|truncated| (truncated, true))
        .unwrap_or_else(|| (data.to_vec(), false))
}

/// Truncate a UTF8 slice to the longest prefix that is still a valid UTF8 string,
/// while being less than `length` bytes and non-empty
fn truncate_utf8(data: &str, length: usize) -> Option<Vec<u8>> {
    let split = (1..=length).rfind(|x| data.is_char_boundary(*x))?;
    Some(data.as_bytes()[..split].to_vec())
}

/// Try and increment the bytes from right to left.
///
/// Returns `None` if all bytes are set to `u8::MAX`.
fn increment(mut data: Vec<u8>) -> Option<Vec<u8>> {
    for byte in data.iter_mut().rev() {
        let (incremented, overflow) = byte.overflowing_add(1);
        *byte = incremented;

        if !overflow {
            return Some(data);
        }
    }

    None
}

/// Try and increment the the string's bytes from right to left, returning when the result
/// is a valid UTF8 string. Returns `None` when it can't increment any byte.
fn increment_utf8(mut data: Vec<u8>) -> Option<Vec<u8>> {
    for idx in (0..data.len()).rev() {
        let original = data[idx];
        let (byte, overflow) = original.overflowing_add(1);
        if !overflow {
            data[idx] = byte;
            if std::str::from_utf8(&data).is_ok() {
                return Some(data);
            }
            data[idx] = original;
        }
    }

    None
}
