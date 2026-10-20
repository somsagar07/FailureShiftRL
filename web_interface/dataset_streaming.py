"""Bounded Parquet reads for Hugging Face row selections."""
from functools import partial


def _parquet_tables(files, row_groups_list=None, *, features, columns, download_config):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets.table import table_cast
    from datasets.utils.file_utils import xopen

    for file_index, item in enumerate(files):
        # datasets 3 used lists of file iterators; datasets 4 uses file paths.
        paths = [item] if isinstance(item, str) else item
        groups = row_groups_list[file_index] if row_groups_list is not None else None
        for path in paths:
            with xopen(path, 'rb', download_config=download_config) as handle:
                reader = pq.ParquetFile(handle, pre_buffer=False)
                try:
                    batches = reader.iter_batches(batch_size=64, row_groups=groups,
                                                  columns=columns, use_threads=False)
                    try:
                        for batch_index, batch in enumerate(batches):
                            table = pa.Table.from_batches([batch])
                            if features is not None:
                                table = table_cast(table, features.arrow_schema)
                            yield (file_index, batch_index), table
                    finally:
                        batches.close()
                finally:
                    reader.close(force=True)


def bounded_parquet_stream(dataset):
    """Keep HF resolution/auth/features while avoiding asynchronous scan cleanup.

    Partially consuming Arrow's Dataset scanner can crash at Python shutdown:
    https://github.com/apache/arrow/issues/45214
    ParquetFile's synchronous batch reader closes before its Python file handle.
    The adapter replaces only this newly loaded stream's table generator; it
    never patches a library class or changes full-dataset loading.
    """
    from datasets import DownloadConfig
    from datasets.packaged_modules.parquet.parquet import Parquet

    examples = getattr(dataset, '_ex_iterable', None)
    builder = getattr(getattr(examples, 'generate_tables_fn', None), '__self__', None)
    if not isinstance(builder, Parquet):
        if getattr(getattr(dataset, 'info', None), 'builder_name', None) == 'parquet':
            raise ValueError('This datasets version uses an unsupported Parquet stream layout. '
                             'Run bash setup.sh to install the supported dependencies.')
        return dataset
    if builder.config.filters is not None:
        raise ValueError('Row-limited loading does not support dataset-defined Parquet filters.')
    examples.generate_tables_fn = partial(
        _parquet_tables, features=builder.info.features, columns=builder.config.columns,
        download_config=DownloadConfig(token=builder.token, storage_options=builder.storage_options))
    return dataset
