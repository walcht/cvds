from __future__ import annotations
from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
from typing import Any, Tuple, Literal
from exceptions import DatasetNotImported
import os
import json
import numpy as np
from dataclasses import dataclass
from metadata_json_encoder import MetadataJSONEncoder


@dataclass
class CVDSMetadata:
    original_dims: tuple[int]
    chunk_size: int
    nbr_chunks_per_resolution_lvl: list[tuple[int]]
    total_nbr_chunks: list[int]
    nbr_resolution_lvls: int
    downsampling_inter: Literal["trilinear"]
    color_depth: int
    lz4_compressed: bool
    decompressed_chunk_size_in_bytes: int
    voxel_dims: tuple[int]
    euler_rotation: tuple[int]


class BaseConverterMetaclass(type, metaclass=ABCMeta):
    """Converter metaclass (i.e., class for creating converter classes). The main purpose of this metaclass
    is to add references to the BaseConverter for every defined class that inherits from BaseConverter.
    """

    def __new__(cls, clsname: str, bases: Tuple[type], namespaces: dict[str, Any]):
        newly_created_cls = super().__new__(cls, clsname, bases, namespaces)
        for base_cls in bases:
            # store reference to this converter class if it inherits from BaseConverter
            if base_cls == BaseConverter:
                BaseConverter.converters.append(newly_created_cls)  # type: ignore
        return newly_created_cls


class BaseConverter(metaclass=BaseConverterMetaclass):
    """Base class for CT (or MRI) datasets convertion to CVDS format"""

    converters: list[type[BaseConverter]] = []

    def __init__(self) -> None:
        self.metadata: CVDSMetadata | None = None

    def write_metadata_stream(
        self,
        text_stream: TextIOWrapper,
    ) -> None:
        """Writes CVDS metadata text to a text (string) stream

        Parameters
        ----------
        text_stream: TextIOWrapper
            string stream to which the CVDS metadata text will be written
        """
        if self.metadata is None:
            raise DatasetNotImported("attempt at writing CVDS metadata before importing a dataset")
        json.dump(self.metadata.__dict__, text_stream, indent=2, cls=MetadataJSONEncoder)

    def write_metadata(
        self,
        output_dir: str,
    ) -> None:
        """Writes CVDS metadata.json file to provided output directory

        Parameters
        ----------
        output_dir : str
            absolute directory path to where the `metadata.json` is going to be written. `output_dir` is expected to be
            empty otherwise an exception will be raised.

        Raises
        ------
        NotADirectoryError
            if provided directory does not exist

        DirectoryNotEmptyError
            if provided directory is not empty
        """
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"provided metadata output directory path: {output_dir} does not exist.")
        with open(os.path.join(output_dir, "metadata.json"), "wt") as ss:
            self.write_metadata_stream(ss)

    @abstractmethod
    def write_binary_chunks(
        self,
        output_dir: str,
    ) -> None:
        """Write CVDS binary chuncks data to the provided directory path

        for each resolution level, a subdirectory is created under the name: resolution_level_<res-lvl> where
        res-lvl is replaced by the resolution level index (e.g., 0 being the highest). The chunks correspondind to that
        resolution level are written in that subdirectory.

        Each chunk file is create under the name: chunk_<chunk-id>.cvds[.lz4] where chunk-id is the id of the chunk
        in the corresponding resolution level. Chunk ID 0 is the upper left most chunk of the first slice. Chunk ID 1 is
        the next chunk along the same row, and so on:

        slice 0:

        +   +   +   +   +   +   +   +   +   +

        +  chunk 0  +  chunk 1  +  chunk 2  +

        +   +   +   +   +   +   +   +   +   +

        +  chunk 3  +  chunk 4  +  chunk 5  +

        +   +   +   +   +   +   +   +   +   +

        +  chunk 6  +  chunk 7  +  chunk 8  +

        +   +   +   +   +   +   +   +   +   +

        slice 1:

        +   +   +   +   +   +   +   +   +   +

        +  chunk 9  +  chunk 10 +  chunk 11 +

        +   +   +   +   +   +   +   +   +   +

        +  chunk 12 +  chunk 13 +  chunk 14 +

        +   +   +   +   +   +   +   +   +   +

        +  chunk 15 +  chunk 16 +  chunk 17 +

        +   +   +   +   +   +   +   +   +   +


        Parameters
        ----------
        output_dir: str
            path to where the binary chunks are going to be written
        """
        ...

    def get_downsampled_dims(
        self,
        original_dims: tuple[int],
        resolution_lvl: int,
    ) -> tuple[int]:
        """Computes the dimensions of the downsampled original volume for the provided resolution level

        Parameters
        ----------
        original_dims: tuple[int]
            original volume dimensions (width, height, depths)
        resolution_lvl : int
            the resolution level for which the downsample dimensions are computed

        Returns
        -------
        tuple[int]
            the downsampled volume dimensions corresponding to this resolution level (x, y, z)
        """
        tmp = np.ceil(np.array(original_dims) / 2**resolution_lvl).astype(np.uint32)
        return (tmp[0].item(), tmp[1].item(), tmp[2].item())

    def get_nbr_chunks(
        self,
        original_dims: tuple[int],
        chunk_size: int,
        resolution_lvl: int,
    ) -> tuple[int]:
        """Computes the number of chunks along each dimension for the provided resolution level

        Parameters
        ----------
        original_dims: tuple[int]
            original volume dimensions (width, height, depths)
        chunk_size: int
            chunk size
        resolution_lvl : int
            the resolution level for which the number of potentially downsampled chunks is computed

        Returns
        -------
        tuple[int]
            the number of chunks corresponding to this resolution level (x, y, z)
        """
        downsampled_dims = np.asarray(self.get_downsampled_dims(original_dims, resolution_lvl), dtype=np.uint32)
        tmp = np.ceil(downsampled_dims / chunk_size).astype(np.uint32)
        return (tmp[0].item(), tmp[1].item(), tmp[2].item())

    def get_max_allowed_resolution_level(
        self,
        original_dims: tuple[int],
        chunk_size: int,
    ) -> int:
        """Computes the maximal allowed resolution level

        the critera to determine tha max allowed resolution level is that along some dimension (x, y, or z) the
        downsampled dimension is less than or equal to chunk_size.

        Parameters
        ----------
        original_dims: tuple[int]
            original volume dimensions (width, height, depths)
        chunk_size: int
            chunk size

        Returns
        -------
        int
            the resolution level up-to and including it which should not be exceeded
        """
        # let x be the maximum resolution level allowed along the dimension X:
        #       dim_x / 2**x <= chunk_size
        #   =>  2**x * chunk_size >= dim_x
        #   =>  x >= log_b2(dim_x / chunk_size)
        # similarly for the dimension Y and Z. Let max_res_lvl be the maximum resolution level allowed along all dimensions:
        #   =>  max_res_lvl = ceil(min(log_b2(dim_x / chunk_size), log_b2(dim_y / chunk_size), ...))
        # the ceil is used because max_res_lvl has to be an integer
        return int(np.ceil(np.min(np.log2(np.array(original_dims) / chunk_size))))

    def get_paddings(
        self,
        original_dims: tuple[int],
        chunk_size: int,
        resolution_lvl: int,
    ) -> tuple[int]:
        """Computes the paddings for the provided resolution level that need to be added to align with chunk size

        Parameters
        ----------
        original_dims: tuple[int]
            original volume dimensions (width, height, depths)
        chunk_size: int
            chunk size
        resolution_lvl : int
            the resolution level for which the paddings are computed

        Returns
        -------
        tuple[int]
            paddings corresponding to this resolution level (x, y, z)
        """
        downsampled_dims = np.array(self.get_downsampled_dims(original_dims, resolution_lvl), dtype=np.uint32)
        nbr_chunks = np.array(self.get_nbr_chunks(original_dims, chunk_size, resolution_lvl), dtype=np.uint32)
        tmp = nbr_chunks * chunk_size - downsampled_dims
        return (tmp[0], tmp[1], tmp[2])

    @abstractmethod
    def import_dataset(
        self,
        dataset_dir: str,
        chunk_size: int = 128,
        nbr_resolution_levels: int = -1,
        lz4_compressed: bool = True,
    ) -> None:
        """Imports the CT (or MRI) dataset to the Chunked Volumetric DataSet (CVDS) format

        Parameters
        ----------
        dataset_dir : str
            absolute path to a dataset directory
        chunk_size : int, optional
            chunk size (i.e., number of voxels per chunk dimension). Should be a power of 2, by default 128
        nbr_resolution_levels : int, optional
            number of resolution levels to be generated up-to and including this resolution_level.
            resolution level 0 corresponds to the original and highest resolution. resolution level 1's chunks cover
            twice the volume covered by resolution level 0's chunks, by default automatically computes the optimal
            number of resolution levels to generate
        lz4_compressed : bool, optional
            whether to compress the chunks using LZ4, by default True
        """
        ...

    @abstractmethod
    def convert_cvds_dataset(
        self,
        cvds_dataset_dir: str,
        output_dir: str,
    ) -> None: ...

    @staticmethod
    @abstractmethod
    def is_this_converter_suitable(
        dataset_dir: str,
    ) -> bool:
        """Checks whether this converter/importer is suitable for the provided CT (or MRI) dataset

        Parameters
        ----------
        dataset_dir : str
            absolute path to a dataset directory

        Returns
        -------
        bool
            True if this converter/importer is suitable. False otherwise
        """
        ...
