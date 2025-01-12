import logging
from converters.converter import BaseConverter
from converters.exceptions import UnsupportedDatasetFormatException


class converter_factory:
    @staticmethod
    def create(dataset_dir_or_fp: str) -> BaseConverter:
        """Factory for creating appropriate converters for a given dataset path.

        Parameters
        ----------
        dataset_dir_or_fp : str
            absolute path to a dataset directory or a single file

        Returns
        -------
        BaseConverter
            suitable converter that can be used to convert the dataset to UVDS format
        """
        logging.debug(f"detected converters: {BaseConverter.converters}")
        for potential_converter in BaseConverter.converters:
            if potential_converter.is_this_converter_suitable(dataset_dir_or_fp):
                logging.debug(f"selected converter: {potential_converter}")
                return potential_converter()
        raise UnsupportedDatasetFormatException(
            f"provided dataset's format at: {dataset_dir_or_fp} is not supported"
        )
