class DatasetNotImported(Exception):
    """Raised when an operation that requires a dataset to be imported is called before importing a dataset"""

    pass


class UnsupportedDatasetFormatException(Exception):
    """Raised when no appropriated converter (or importer) for a provided dataset format is found"""

    pass


class DirectoryNotEmptyError(Exception):
    """Raised when a directory path is not empty"""

    pass
