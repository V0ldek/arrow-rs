use crate::compression::{self, Codec, CodecOptionsBuilder};
use crate::format as parquet;
use crate::format::{ColumnIndex, OffsetIndex};
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
use thrift::protocol::TCompactOutputProtocol;

use crate::column::writer::{
    get_typed_column_writer_mut, ColumnCloseResult, ColumnMetrics, ColumnWriterImpl,
};
use crate::column::{
    page::{CompressedPage, Page, PageWriteSpec, PageWriter},
    writer::{get_column_writer, ColumnWriter},
};
use crate::data_type::DataType;
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

        let statistics_enabled = props.statistics_enabled(column.path());
        let column_metrics = ColumnMetrics::new();

        Ok(Self {
            descr: column,
            sink: buf,
            on_close,
            props,
            codec,
            compressor,
            statistics_enabled,
            column_metrics,
        })
    }

    pub fn write_decoder(
        &mut self,
        decoder_bytes: &[u8],
        version: impl Into<String>,
    ) -> Result<()> {
        let uncompressed_size = decoder_bytes.len();

        let buf = if let Some(ref mut cmpr) = self.compressor {
            let mut compressed_buf = Vec::with_capacity(uncompressed_size);
            cmpr.compress(decoder_bytes, &mut compressed_buf)?;
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

    pub fn write_data_page(
        &mut self,
        page_data: &[u8],
        min: Option<T>,
        max: Option<T>,
        num_nulls: u32,
        num_values: u32,
        num_distinct: Option<u64>,
        uncompressed_len: usize,
    ) -> Result<()> {
        let buf = Bytes::copy_from_slice(page_data);

        self.column_metrics.total_rows_written += u64::from(num_values);
        self.column_metrics.num_column_nulls += u64::from(num_nulls);
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
        let statistics = Statistics::new(min, max, num_distinct, Some(num_nulls.into()), false);

        let page = Page::DataPageV2 {
            buf,
            num_values,
            encoding: crate::basic::Encoding::IGNITION,
            num_nulls,
            num_rows: num_values,
            def_levels_byte_len: 0,
            rep_levels_byte_len: 0,
            is_compressed: true,
            statistics: Some(statistics),
        };
        let compressed_page = CompressedPage::new(page, uncompressed_len);

        let page_spec = self.write_page(compressed_page)?;
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

        let result = ColumnCloseResult {
            bytes_written: self.column_metrics.total_bytes_written,
            rows_written: self.column_metrics.total_rows_written,
            bloom_filter,
            metadata,
            column_index: None,
            offset_index: None,
        };
        println!("Column close: {result:?}");
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
