from converter import BaseConverter, CVDSMetadata
from dataclasses import dataclass
import os
import glob
import logging
import numpy as np
import cv2 as cv
from numpy.typing import NDArray
import click
from tqdm import trange
from typing import Any
import json
import lz4.block


@dataclass
class ImagesMetadata:
    """original volume dimensions (x, y, z) or (width, height, depth)"""

    original_dims: tuple[int]
    """color depth - either 8bit or 16bit"""
    color_depth: int
    converted_to_8bit: bool


class ImagesConverter(BaseConverter):
    """Imports and converts an image sequence into a CVDS. Images are expected to be ordered according to their
    filenames. Each image represents a 2D slice of the volumetric data along some axis. Image files do not contain
    volume attributes such as width, height or depth, therefore command line prompt the user for these values.

    Supported image file formats are:
        - TIFF (.tif)
        - PNG (.png)
    """

    supported_formats: dict[str, str] = {"tiff": ".tif", "png": ".png", "bmp": ".bmp"}

    def __init__(self) -> None:
        super().__init__()
        self.sorted_image_fps: list[str]

    def get_sorted_images_from_dir(
        self,
        dataset_dir: str,
        verbose: bool = True,
    ) -> list[str]:
        for supported_format, ext in self.supported_formats.items():
            fps = [fp for fp in glob.glob(os.path.join(dataset_dir, f"*{ext}"), recursive=False)]
            if fps:
                if verbose:
                    print(f"found: {len(fps)} {supported_format.upper()} image files")
                # order according to filename - very important!
                return sorted(fps)
        if verbose:
            print("could not find any supported image files")
        return []

    def extract_images_metadata(
        self,
        sorted_image_fps: list[str],
        force_8bit_conversion=True,
        verbose: bool = True,
    ) -> ImagesMetadata:
        dims: NDArray[np.uint32]
        color_depth: np.dtype
        if force_8bit_conversion:
            img = cv.imread(sorted_image_fps[0], cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(sorted_image_fps[0], cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
        dims = (img.shape[1], img.shape[0], len(sorted_image_fps))
        img_dtype = img.dtype
        if img_dtype == np.uint8:
            color_depth = 8
        elif img_dtype == np.uint16:
            color_depth = 16
        else:
            raise ValueError(f"unsupported image color depth: {img_dtype}")
        # verify that all provided images are consistent
        for fp in sorted_image_fps:
            if force_8bit_conversion:
                img = cv.imread(fp, cv.IMREAD_GRAYSCALE)
            else:
                img = cv.imread(fp, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
            if img.dtype != img_dtype:
                raise ValueError(f"inconsistent image color depth across volume image slice(s): {fp}")
            if (img.shape[1] != dims[0]) or (img.shape[0] != dims[1]):
                raise ValueError(f"inconsistent image dimension(s) across volume image slice(s): {fp}")
        if verbose:
            print(f"volume color depth: {color_depth}")
            print(f"original volume dimensions (x, y, z): ({dims[0]},{dims[1]},{dims[2]})")
        return ImagesMetadata(original_dims=dims, color_depth=color_depth, converted_to_8bit=force_8bit_conversion)

    def import_dataset(
        self,
        dataset_dir: str,
        chunk_size: int = 128,
        nbr_resolution_levels: int = -1,
        lz4_compressed: bool = True,
        force_8bit_conversion: bool = True,
    ) -> None:
        if not os.path.isdir(dataset_dir):
            raise NotADirectoryError(f"provided images path is not a directory path: ${dataset_dir}")
        self.sorted_image_fps = self.get_sorted_images_from_dir(dataset_dir)
        if not self.sorted_image_fps:
            raise RuntimeError("no images of any supported formats is found in the provided directory")
        metadata = self.extract_images_metadata(self.sorted_image_fps, force_8bit_conversion)
        max_res_lvl = self.get_max_allowed_resolution_level(metadata.original_dims, chunk_size)
        nbr_resolution_levels = min(nbr_resolution_levels, max_res_lvl)
        if nbr_resolution_levels < 0:
            nbr_resolution_levels = max_res_lvl + 1
        nbr_chunks_per_res_lvl = [
            self.get_nbr_chunks(metadata.original_dims, chunk_size, res_lvl)
            for res_lvl in range(0, nbr_resolution_levels)
        ]
        total_nbr_chunks = [x * y * z for x, y, z in nbr_chunks_per_res_lvl]
        decompressed_chunk_size_in_bytes: int
        if metadata.color_depth == 8:
            decompressed_chunk_size_in_bytes = chunk_size**3
        elif metadata.color_depth == 16:
            decompressed_chunk_size_in_bytes = (chunk_size**3) * 2
        else:
            raise ValueError(f"unknown metadata color depth value: {metadata.color_depth}")
        # prompt the user for additional, non-available information:
        voxel_dim_x = click.prompt("enter voxel dimension along X axis [mm] ", type=float, default=1.0)
        voxel_dim_y = click.prompt("enter voxel dimension along Y axis [mm] ", type=float, default=1.0)
        voxel_dim_z = click.prompt("enter voxel dimension along Z axis [mm] ", type=float, default=1.0)
        euler_rot_x = click.prompt("enter Euler rotation around X axis [°]  ", type=float, default=180)
        euler_rot_y = click.prompt("enter Euler rotation around Y axis [°]  ", type=float, default=0.0)
        euler_rot_z = click.prompt("enter Euler rotation around Z axis [°]  ", type=float, default=0.0)
        self.metadata = CVDSMetadata(
            original_dims=metadata.original_dims,
            chunk_size=chunk_size,
            color_depth=metadata.color_depth,
            force_8bit_conversion=force_8bit_conversion,
            nbr_resolution_lvls=nbr_resolution_levels,
            nbr_chunks_per_resolution_lvl=nbr_chunks_per_res_lvl,
            total_nbr_chunks=total_nbr_chunks,
            lz4_compressed=lz4_compressed,
            decompressed_chunk_size_in_bytes=decompressed_chunk_size_in_bytes,
            downsampling_inter="trilinear",
            voxel_dims=(voxel_dim_x, voxel_dim_y, voxel_dim_z),
            euler_rotation=(euler_rot_x, euler_rot_y, euler_rot_z),
        )

    def convert_cvds_dataset(
        self,
        cvds_dataset_dir: str,
        output_dir: str,
        resolution_lvl: int = 0,
        format: str = "PNG",
    ) -> None:
        """Converts a given CVDS dataset to a set of images. This is mainly
        used for debugging/testing purposes

        Parameters
        ----------
        cvds_dataset_dir : str
            path to a CVDS dataset root directory (contains metadata.json and resolution-level subdirectories)
        output_dir : str
            output directory path. Images are ordered by their filenames.
        resolution_lvl: int, optional
            the resolution level to export, by default 0
        format: str, optional
            the image format to use, by default "PNG". Note that certain formats may not support writing 16-bit
            uint grayscale images.
        """
        if not os.path.isdir(cvds_dataset_dir):
            raise NotADirectoryError(f"CVDS dataset path is not a directory path: ${cvds_dataset_dir}")
        res_lvl_chunks_path = os.path.join(cvds_dataset_dir, f"resolution_level_{resolution_lvl}")
        if not os.path.isdir(res_lvl_chunks_path):
            raise NotADirectoryError(
                f"CVDS dataset path doesn't contain resolution level subdirectory: {res_lvl_chunks_path}"
            )
        metadata_fp = os.path.join(cvds_dataset_dir, "metadata.json")
        metadata: CVDSMetadata
        with open(metadata_fp, mode="rt") as ts:
            try:
                metadata = CVDSMetadata(**json.load(ts))
            except json.JSONDecodeError:
                return
        if resolution_lvl > metadata.nbr_resolution_lvls:
            raise Exception(f"provided resolution level is out of range: {resolution_lvl}")
        for slice_id in trange(
            metadata.nbr_chunks_per_resolution_lvl[resolution_lvl][2] * metadata.chunk_size,
            desc=f"converting CVDS to images [res={resolution_lvl}]",
            unit="image",
        ):
            nbr_chunks_x = metadata.nbr_chunks_per_resolution_lvl[resolution_lvl][0]
            nbr_chunks_y = metadata.nbr_chunks_per_resolution_lvl[resolution_lvl][1]
            dim_x = nbr_chunks_x * metadata.chunk_size
            dim_y = nbr_chunks_y * metadata.chunk_size
            img_dtype: np.dtype
            bpp: int  # bytes per pixel
            if metadata.color_depth == 8:
                img_dtype = np.uint8
                bpp = 1
            elif metadata.color_depth == 16:
                img_dtype = np.uint16
                bpp = 2
            else:
                raise ValueError(f"unsupported color depth value: {metadata.color_depth}")
            img_data = np.ndarray((dim_y, dim_x), dtype=img_dtype)
            for i in range(nbr_chunks_x * nbr_chunks_y):
                chunk_id = i + nbr_chunks_x * nbr_chunks_y * (slice_id // metadata.chunk_size)
                chunk_fp: str
                chunk_slice: np.memmap
                offset = (slice_id % metadata.chunk_size) * metadata.chunk_size * metadata.chunk_size * bpp
                if metadata.lz4_compressed:
                    chunk_fp = os.path.join(res_lvl_chunks_path, f"chunk_{chunk_id}.cvds.lz4")
                    decompressed_data: bytes
                    with open(chunk_fp, mode="rb") as bs:
                        decompressed_data = lz4.block.decompress(bs.read())
                    chunk_slice = np.frombuffer(
                        decompressed_data,
                        img_dtype,
                        count=metadata.chunk_size * metadata.chunk_size * bpp,  # TODO: verify this
                        offset=offset,
                    ).reshape((metadata.chunk_size, metadata.chunk_size))
                else:
                    chunk_fp = os.path.join(res_lvl_chunks_path, f"chunk_{chunk_id}.cvds")
                    chunk_slice = np.memmap(
                        chunk_fp, img_dtype, "r", offset, (metadata.chunk_size, metadata.chunk_size)
                    )
                row_start_idx = metadata.chunk_size * (i // nbr_chunks_x)
                row_end_idx = row_start_idx + metadata.chunk_size
                col_start_idx = metadata.chunk_size * (i % nbr_chunks_x)
                col_end_idx = col_start_idx + metadata.chunk_size
                img_data[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = chunk_slice
            img_fp: str = os.path.join(output_dir, f"{slice_id}.{format.lower()}")
            cv.imwrite(img_fp, img_data)

    def _post_processing(
        self,
        output_dir: str,
    ):
        # do nothing LZ4 compression is disabled
        if not self.metadata.lz4_compressed:
            return
        logging.info("starting post processing step(s) ...")
        for res_lvl in range(0, self.metadata.nbr_resolution_lvls):
            res_lvl_dir = os.path.join(output_dir, f"resolution_level_{res_lvl}")
            for chunk_id in trange(
                0,
                self.metadata.total_nbr_chunks[res_lvl],
                desc=f"compressing binary chunks for res lvl: {res_lvl}",
                unit="chunks",
            ):
                uncompressed_fp = os.path.join(res_lvl_dir, f"chunk_{chunk_id}.cvds")
                compressed_fp = os.path.join(res_lvl_dir, f"chunk_{chunk_id}.cvds.lz4")
                with open(uncompressed_fp, mode="rb") as uncompressed_bs:
                    with open(compressed_fp, mode="wb") as compressed_bs:
                        compressed_bs.write(
                            lz4.block.compress(
                                uncompressed_bs.read(),
                                mode="high_compression",
                                # make sure this is set to False otherwise C# side will not work!
                                store_size=False,
                            )
                        )
                # don't forget to remove the uncompressed chunk
                os.remove(uncompressed_fp)
                pass
        logging.info("post processing done")

    def _write_slice(
        self,
        data: NDArray[Any],
        nbr_chunks_x: int,
        nbr_chunks_y: int,
        nbr_written_slices: int,
        resolution_lvl_dir: str,
    ) -> None:
        for i in range(nbr_chunks_x * nbr_chunks_y):
            chunk_id = i + (nbr_written_slices // self.metadata.chunk_size) * nbr_chunks_x * nbr_chunks_y
            fp = os.path.join(resolution_lvl_dir, f"chunk_{chunk_id}.cvds")
            with open(fp, mode="ab") as binary_stream:
                row_start_idx = self.metadata.chunk_size * (i // nbr_chunks_x)
                row_end_idx = row_start_idx + self.metadata.chunk_size
                col_start_idx = self.metadata.chunk_size * (i % nbr_chunks_x)
                col_end_idx = col_start_idx + self.metadata.chunk_size
                binary_stream.write(data[row_start_idx:row_end_idx, col_start_idx:col_end_idx].tobytes())

    def write_binary_chunks(
        self,
        output_dir: str,
    ) -> None:
        # read one image at a time
        if self.sorted_image_fps is None or not len(self.sorted_image_fps):
            raise RuntimeError("attempt at writing binary chunks before importing a dataset")
        if not os.path.isdir(output_dir):
            raise NotADirectoryError(f"provided output directory is, in fact, not a directory: {output_dir}")
        original_dims = self.metadata.original_dims
        chunk_size = self.metadata.chunk_size
        dtype: np.dtype
        if self.metadata.color_depth == 8:
            dtype = np.uint8
        elif self.metadata.color_depth == 16:
            dtype = np.uint16
        else:
            raise TypeError(f"unsupported image color depth: {self.metadata.color_depth}")
        for res_lvl in range(0, self.metadata.nbr_resolution_lvls):
            # create dir containing chunks for this resolution level
            res_lvl_dir = os.path.join(output_dir, f"resolution_level_{res_lvl}")
            os.mkdir(res_lvl_dir)
            dims = self.get_downsampled_dims(original_dims, res_lvl)
            nbr_chunks = self.get_nbr_chunks(original_dims, chunk_size, res_lvl)
            # Note: it is crucial that paddding is added per-resolution level (as opposed to being added once at highest
            # resolution level then downsampling)
            pads = self.get_paddings(original_dims, chunk_size, res_lvl)
            averaged_slices_data: NDArray[np.float32] | None = None
            nbr_written_slices: int = 0
            for slice_idx in trange(
                len(self.sorted_image_fps), desc=f"writing chunks for resolution level {res_lvl}", unit="slices"
            ):
                # read and downsample current slice
                if self.metadata.force_8bit_conversion:
                    img = cv.imread(self.sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE)
                else:
                    img = cv.imread(self.sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
                slice_data = cv.resize(img, (dims[0], dims[1]), 0, 0, cv.INTER_LINEAR)
                # add padding (note that we add padding after downsampling to avoid averaging with padded values!)
                padded_slice_data = np.pad(
                    slice_data, pad_width=((0, pads[1]), (0, pads[0])), mode="constant", constant_values=(0,)
                ).astype(np.float32) / (2**res_lvl)
                # Example: if res_lvl == 1, then each two successive slices should be downsampled once, averaged (here
                # it means divided by 2 (2**res_lvl == 2)) then summed with the averaged_slices_data array. Finally the
                # resulting data should be written to the corresponding chunks for that resolution level.
                if (slice_idx % 2**res_lvl) == 0:
                    # average and write the previous averaged slices data
                    if averaged_slices_data is not None:
                        # convert back to original dtype
                        data = averaged_slices_data.astype(dtype)
                        self._write_slice(data, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir)
                        nbr_written_slices += 1
                    # overwrite/initialize the averaged slices data
                    averaged_slices_data = np.zeros(
                        (nbr_chunks[1] * self.metadata.chunk_size, nbr_chunks[0] * self.metadata.chunk_size),
                        dtype=np.float32,
                    )
                averaged_slices_data += padded_slice_data
            # don't forget to write the last slice!
            data = averaged_slices_data.astype(dtype)
            self._write_slice(data, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir)
            nbr_written_slices += 1
            # finally, write the zero-padding slices
            zeros = np.zeros((nbr_chunks[1] * chunk_size, nbr_chunks[0] * chunk_size), dtype=dtype)
            for _ in range(pads[2]):
                self._write_slice(zeros, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir)
                nbr_written_slices += 1
            assert nbr_written_slices == nbr_chunks[2] * chunk_size, (
                f"unexpected number of written slice for the resolution level: {res_lvl}"
            )
        # finally, perform post processing (e.g., to compress the chunks because stream compression is a bitch)
        self._post_processing(output_dir)

    def write_histogram(
        self,
        output_dir: str,
        resolution: int = 1024,
    ) -> np.ndarray:
        hist = np.zeros((resolution,), dtype=np.uint64)
        for slice_idx in trange(len(self.sorted_image_fps), desc="reading image slice", unit="slice"):
            img = cv.imread(self.sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE)
            np.add(
                np.histogram(
                    img, bins=resolution, range=(np.iinfo(np.uint8).min, np.iinfo(np.uint8).max), density=False
                )[0].astype(np.uint64),
                hist,
                out=hist,
            )

        with open(os.path.join(output_dir, "histogram.bin"), mode="wb") as bs:
            bs.write(hist.tobytes())

    @staticmethod
    def is_this_converter_suitable(
        dataset_dir: str,
    ) -> bool:
        if os.path.isdir(dataset_dir):
            for supported_format, ext in ImagesConverter.supported_formats.values():
                fps = glob.glob(os.path.join(dataset_dir, f"*{ext}"), recursive=False)
                if fps:
                    logging.info(f"found: {len(fps)} of images of the supported format: {supported_format.upper()}")
                    return True
        return False


if __name__ == "__main__":
    converter = ImagesConverter()
    converter.import_dataset(
        dataset_dir=r"C:\Users\walid\Desktop\thesis_test_datasets\Fish_200MB\slices",
        chunk_size=256,
        lz4_compressed=True,
        force_8bit_conversion=True,
    )
    # converter.write_metadata(r"C:\Users\walid\Desktop\chunk_size_param_stats\Fish 256 LZ4")
    # converter.write_binary_chunks(r"C:\Users\walid\Desktop\chunk_size_param_stats\Fish 256 LZ4")
    converter.write_histogram(output_dir=".")

    # converter.convert_cvds_dataset(
    #     r"C:\Users\walid\Desktop\test_dataset_compressed", output_dir="tmp", resolution_lvl=0
    # )
    print("done")
