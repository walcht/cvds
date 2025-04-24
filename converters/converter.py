from __future__ import annotations
from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
from typing import Any, Tuple, Literal
from exceptions import DatasetNotImported
import os
import json
import numpy as np
import numpy.typing as npt
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
    force_8bit_conversion: bool
    lz4_compressed: bool
    decompressed_chunk_size_in_bytes: int
    vdhms: list[tuple[int, float]]
    vdhm_penalty: float
    octree_nrb_nodes: int
    octree_max_depth: int
    octree_smallest_subdivision: list[int]
    octree_size_in_bytes: int
    histogram_nbr_bins: int
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
        self.output_dir: str

    def _write_metadata_stream(self, text_stream: TextIOWrapper) -> None:
        """Writes CVDS metadata text to a text (string) stream

        Parameters
        ----------
        text_stream: TextIOWrapper
            string stream to which the CVDS metadata text will be written
        """

        if self.metadata is None:
            raise DatasetNotImported("attempt at writing CVDS metadata before importing a dataset")
        json.dump(self.metadata.__dict__, text_stream, indent=2, cls=MetadataJSONEncoder)

    def _write_metadata(self) -> None:
        """Writes CVDS metadata.json file to provided output directory

        Raises
        ------
        NotADirectoryError
            if provided directory does not exist

        DirectoryNotEmptyError
            if provided directory is not empty
        """
        
        if not os.path.isdir(self.output_dir):
            raise NotADirectoryError(f"provided metadata output directory path: {self.output_dir} does not exist.")
        with open(os.path.join(self.output_dir, "metadata.json"), "wt") as ss:
            self._write_metadata_stream(ss)

    def _get_downsampled_dims(
        self,
        original_dims: npt.NDArray,
        resolution_lvl: int,
    ) -> npt.NDArray:
        """Computes the dimensions of the downsampled original volume for the provided resolution level

        Parameters
        ----------
        original_dims: npt.NDArray
            original volume dimensions (width, height, depths)

        resolution_lvl : int
            the resolution level for which the downsample dimensions are computed

        Returns
        -------
        npt.NDArray
            the downsampled volume dimensions corresponding to this resolution level (x, y, z)
        """
        return np.ceil(original_dims / 2**resolution_lvl).astype(np.uint32)

    def _get_nbr_chunks(
        self,
        original_dims: npt.NDArray,
        chunk_size: int,
        resolution_lvl: int,
    ) -> npt.NDArray:
        """Computes the number of chunks along each dimension for the provided resolution level

        Parameters
        ----------
        original_dims: npt.NDArray
            original volume dimensions (width, height, depths)

        chunk_size: int
            chunk size

        resolution_lvl : int
            the resolution level for which the number of potentially downsampled chunks is computed

        Returns
        -------
        npt.NDArray
            the number of chunks corresponding to this resolution level (x, y, z)
        """
        downsampled_dims = self._get_downsampled_dims(original_dims, resolution_lvl)
        return np.ceil(downsampled_dims / chunk_size).astype(np.uint32)

    def _get_max_allowed_resolution_level(
        self,
        original_dims: npt.NDArray,
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
        return int(np.ceil(np.min(np.log2(original_dims / chunk_size))))

    def _get_paddings(
        self,
        original_dims: npt.NDArray,
        chunk_size: int,
        resolution_lvl: int,
    ) -> npt.NDArray:
        """Computes the paddings for the provided resolution level that need to be added to align with chunk size

        Parameters
        ----------
        original_dims: npt.NDArray
            original volume dimensions (width, height, depths)

        chunk_size: int
            chunk size

        resolution_lvl : int
            the resolution level for which the paddings are computed

        Returns
        -------
        npt.NDArray
            paddings corresponding to this resolution level (x, y, z)
        """
        downsampled_dims = self._get_downsampled_dims(original_dims, resolution_lvl)
        nbr_chunks = self._get_nbr_chunks(original_dims, chunk_size, resolution_lvl)
        return nbr_chunks * chunk_size - downsampled_dims

    @abstractmethod
    def import_dataset(
        self,
        dataset_dir: str,
        output_dir: str,
        chunk_size: int = 128,
        nbr_resolution_levels: int = -1,
        lz4_compressed: bool = True,
        force_8bit_conversion: bool = True,
        vdhm_tolerance_range: tuple[int, int] = (0, 10),
    ) -> None:
        """Imports the CT (or MRI) dataset to the Chunked Volumetric DataSet (CVDS) format

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
        dataset_dir : str
            absolute path to a dataset directory

        output_dir : str
            absolute directory path to where the `metadata.json`, chunks in different resolution levels, the residency
            octree, and the histogram are going to be written. `output_dir` is expected to be empty otherwise an
            exception will be raised.

        chunk_size : int, optional
            chunk size (i.e., number of voxels per chunk dimension). Should be a power of 2, by default 128

        nbr_resolution_levels : int, optional
            number of resolution levels to be generated up-to and including this resolution_level.
            resolution level 0 corresponds to the original and highest resolution. resolution level 1's chunks cover
            twice the volume covered by resolution level 0's chunks, by default automatically computes the optimal
            number of resolution levels to generate

        lz4_compressed : bool, optional
            whether to compress the chunks using LZ4, by default True

        force_8bit_conversion : bool, optional
            if set, converts non-8-bit input datasets into a color depth of 8 bits, by default True

        vdhm_tolerance_range : tuple[int, int], optional
            for each VDHM tolerance value in this range, VDHM is measured for that tolerance, by default (0, 10)
        """
        ...

    @abstractmethod
    def convert_cvds_dataset(
        self,
        cvds_dataset_dir: str,
        output_dir: str,
        resolution_lvl: int = 0,
        /,
        **kwargs,
    ) -> None:
        """Converts a provided CVDS dataset into this convertes format. For example, for images_converter this is
        converts CVDS to image slices. This is mainly used for debugging and testing purposes."""
        ...

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
