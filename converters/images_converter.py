from io import TextIOWrapper, BufferedWriter
from converter import BaseConverter, CVDSMetadata
from dataclasses import dataclass
import os
from PIL import Image
import glob
import logging
import numpy as np
from numpy.typing import NDArray
import click
import lz4.frame
from tqdm import trange
from typing import Any


@dataclass
class ImagesMetadata:
    original_dims: tuple[int]
    color_depth: int


class ImagesConverter(BaseConverter):
    """Imports and converts an image sequence into a CVDS. Images are expected to be ordered according to their
    filenames. Each image represents a 2D slice of the volumetric data along some axis. Image files do not contain
    volume attributes such as width, height or depth, therefore command line prompt the user for these values.

    Supported image file formats are:
        - TIFF (.tif)
        - PNG (.png)
    """

    supported_formats: dict[str, str] = {"TIFF": ".tif", "PNG": ".png"}

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

    def extract_images_metadata(self, sorted_image_fps: list[str], verbose: bool = True) -> ImagesMetadata:
        dims: NDArray[np.uint32]
        color_depth: np.dtype
        img_mode: str
        with Image.open(sorted_image_fps[0]) as img:
            dims = (img.width, img.height, len(sorted_image_fps))
            img_mode = img.mode
            if img.mode in {"L", "P"}:
                color_depth = 8
            elif img.mode == "I;16":
                color_depth = 16
            else:
                raise ValueError(f"unsupported image mode/format: {img.mode}")
        # verify that all provided images are consistent
        for fp in sorted_image_fps:
            with Image.open(fp) as img:
                if img.mode != img_mode:
                    raise ValueError(f"inconsistent image mode across volume image slice(s): {fp}")
                if (img.width != dims[0]) or (img.height != dims[1]):
                    raise ValueError(f"inconsistent image dimension(s) across volume image slice(s): {fp}")
        if verbose:
            print(f"volume color depth: {color_depth}")
            print(f"original volume dimensions (x, y, z): ({dims[0]},{dims[1]},{dims[2]})")
        return ImagesMetadata(original_dims=dims, color_depth=color_depth)

    def import_dataset(
        self,
        dataset_dir: str,
        chunk_size: int = 128,
        nbr_resolution_levels: int = -1,
        lz4_compressed: bool = True,
    ) -> None:
        if not os.path.isdir(dataset_dir):
            raise NotADirectoryError(f"provided images path is not a directory path: ${dataset_dir}")
        self.sorted_image_fps = self.get_sorted_images_from_dir(dataset_dir)
        if not self.sorted_image_fps:
            raise RuntimeError("no images of any supported formats is found in the provided directory")
        metadata = self.extract_images_metadata(self.sorted_image_fps)
        max_res_lvl = self.get_max_allowed_resolution_level(metadata.original_dims, chunk_size)
        nbr_resolution_levels = min(nbr_resolution_levels, max_res_lvl)
        if nbr_resolution_levels < 0:
            nbr_resolution_levels = max_res_lvl
        nbr_chunks_per_res_lvl = [
            self.get_nbr_chunks(metadata.original_dims, chunk_size, res_lvl)
            for res_lvl in range(0, nbr_resolution_levels + 1)
        ]
        total_nbr_chunks = [x + y + z for x, y, z in nbr_chunks_per_res_lvl]
        # prompt the user for additional, non-available information:
        voxel_dim_x = click.prompt("enter voxel dimension along X axis [default=1.0] [mm]: ", type=float, default=1.0)
        voxel_dim_y = click.prompt("enter voxel dimension along Y axis [default=1.0] [mm]: ", type=float, default=1.0)
        voxel_dim_z = click.prompt("enter voxel dimension along Z axis [default=1.0] [mm]: ", type=float, default=1.0)
        euler_rot_x = click.prompt("enter Euler rotation around X axis [default=0.0]  [°]: ", type=float, default=0.0)
        euler_rot_y = click.prompt("enter Euler rotation around Y axis [default=0.0]  [°]: ", type=float, default=0.0)
        euler_rot_z = click.prompt("enter Euler rotation around Z axis [default=0.0]  [°]: ", type=float, default=0.0)
        self.metadata = CVDSMetadata(
            original_dims=metadata.original_dims,
            chunk_size=chunk_size,
            color_depth=metadata.color_depth,
            nbr_resolution_lvls=nbr_resolution_levels,
            nbr_chunks_per_resolution_lvl=nbr_chunks_per_res_lvl,
            total_nbr_chunks=total_nbr_chunks,
            lz4_compressed=lz4_compressed,
            downsampling_inter="trilinear",
            voxel_dims=(voxel_dim_x, voxel_dim_y, voxel_dim_z),
            euler_rotation=(euler_rot_x, euler_rot_y, euler_rot_z),
        )

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
            try:
                binary_stream: TextIOWrapper | BufferedWriter
                if self.metadata.lz4_compressed:
                    fp = os.path.join(resolution_lvl_dir, f"chunk_{chunk_id}.cvds.lz4")
                    binary_stream = lz4.frame.open(fp, mode="ab", compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
                else:
                    fp = os.path.join(resolution_lvl_dir, f"chunk_{chunk_id}.cvds")
                    binary_stream = open(fp, mode="ab")
                row_start_idx = self.metadata.chunk_size * (i // nbr_chunks_x)
                row_end_idx = row_start_idx + self.metadata.chunk_size
                col_start_idx = self.metadata.chunk_size * (i % nbr_chunks_x)
                col_end_idx = col_start_idx + self.metadata.chunk_size
                binary_stream.write(data[row_start_idx:row_end_idx, col_start_idx:col_end_idx].tobytes())
            finally:
                binary_stream.close()

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
        for res_lvl in range(0, self.metadata.nbr_resolution_lvls + 1):
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
                len(self.sorted_image_fps), desc=f"writing chunks for resolution level {res_lvl}", unit="slice"
            ):
                # read and downsample current slice
                slice_data: NDArray[Any]
                with Image.open(self.sorted_image_fps[slice_idx]) as img:
                    if res_lvl > 0:
                        slice_data = np.asarray(img.resize((dims[0], dims[1]), Image.Resampling.BILINEAR))
                    else:
                        slice_data = np.asarray(img)
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
        dataset_dir=r"C:\Users\walid\Desktop\CTDatasets\dataset_02",
        chunk_size=128,
        lz4_compressed=True,
    )
    converter.write_metadata(r"output")
    # converter.write_binary_chunks(r"output")
    print("done")
