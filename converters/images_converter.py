if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from converter import BaseConverter, CVDSMetadata
from dataclasses import dataclass
from common import octree_utils
import glob
import logging
import numpy as np
import numpy.typing as npt
import cv2 as cv
import click
from tqdm import trange
import json
import lz4.block


@dataclass
class ImagesMetadata:
    """original volume dimensions (x, y, z) or (width, height, depth)"""

    original_dims: npt.NDArray
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
        - BMP (.bmp)
    """

    supported_formats: dict[str, str] = {"tiff": ".tif", "png": ".png", "bmp": ".bmp"}

    def __init__(self) -> None:
        super().__init__()
        self.sorted_image_fps: list[str]

    def _get_sorted_images_from_dir(
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

    def _extract_images_metadata(
        self,
        sorted_image_fps: list[str],
        force_8bit_conversion=True,
        verbose: bool = True,
    ) -> ImagesMetadata:
        color_depth: np.dtype
        if force_8bit_conversion:
            img = cv.imread(sorted_image_fps[0], cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(sorted_image_fps[0], cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
        dims = np.array([img.shape[1], img.shape[0], len(sorted_image_fps)], dtype=np.uint32)
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
        output_dir: str,
        chunk_size: int = 128,
        nbr_resolution_levels: int = -1,
        lz4_compressed: bool = True,
        force_8bit_conversion: bool = True,
        octree_min_nbr_voxels: int = 16,
        vdhm_tolerance_range: tuple[int, int] = (0, 10),
        vdhm_penalty: float = 8.001,
        histogram_nbr_bins: int = 1024,
    ) -> None:
        if not os.path.isdir(dataset_dir):
            raise NotADirectoryError(f"provided images path is not a directory path: ${dataset_dir}")
        self.sorted_image_fps = self._get_sorted_images_from_dir(dataset_dir)
        if not self.sorted_image_fps:
            raise RuntimeError("no images of any supported formats is found in the provided directory")

        # prompt the user for additional, non-available information:
        voxel_dim_x = click.prompt("enter voxel dimension along X axis [mm] ", type=float, default=1.0)
        voxel_dim_y = click.prompt("enter voxel dimension along Y axis [mm] ", type=float, default=1.0)
        voxel_dim_z = click.prompt("enter voxel dimension along Z axis [mm] ", type=float, default=1.0)
        euler_rot_x = click.prompt("enter Euler rotation around X axis [°]  ", type=float, default=180)
        euler_rot_y = click.prompt("enter Euler rotation around Y axis [°]  ", type=float, default=0.0)
        euler_rot_z = click.prompt("enter Euler rotation around Z axis [°]  ", type=float, default=0.0)

        self.output_dir = output_dir
        imgs_metadata = self._extract_images_metadata(self.sorted_image_fps, force_8bit_conversion)
        max_res_lvl = self._get_max_allowed_resolution_level(imgs_metadata.original_dims, chunk_size)

        # compute or set number of resoution levels to generate
        nbr_resolution_levels = min(nbr_resolution_levels, max_res_lvl)
        if nbr_resolution_levels < 0:
            nbr_resolution_levels = max_res_lvl + 1

        nbr_chunks_per_res_lvl = [
            self._get_nbr_chunks(imgs_metadata.original_dims, chunk_size, res_lvl).tolist()
            for res_lvl in range(0, nbr_resolution_levels)
        ]
        total_nbr_chunks = [x * y * z for x, y, z in nbr_chunks_per_res_lvl]

        # determine original color depth
        decompressed_chunk_size_in_bytes: int
        dtype: np.dtype
        if imgs_metadata.color_depth == 8:
            dtype = np.uint8
            decompressed_chunk_size_in_bytes = chunk_size**3
        elif imgs_metadata.color_depth == 16:
            dtype = np.uint16
            decompressed_chunk_size_in_bytes = (chunk_size**3) * 2
        else:
            raise ValueError(f"unknown metadata color depth value: {imgs_metadata.color_depth}")

        # write binary chunks, residency octree, and the histogram
        if self.sorted_image_fps is None or not len(self.sorted_image_fps):
            raise RuntimeError("attempt at writing binary chunks before importing a dataset")
        if not os.path.isdir(self.output_dir):
            raise NotADirectoryError(f"provided output directory is, in fact, not a directory: {self.output_dir}")

        volume_dims = imgs_metadata.original_dims

        # residency octree data structure - we write to it in here to avoid a very costly separate function call
        residency_octree, max_octree_depth = octree_utils.create_and_initialize_octree(
            volume_dims, octree_min_nbr_voxels
        )

        # histogram container
        hist = np.zeros((histogram_nbr_bins,), dtype=np.uint64)

        # TODO: make this the inner loop - this will significantly improve conversion time!
        for res_lvl in range(0, nbr_resolution_levels):
            # create dir containing chunks for this resolution level
            res_lvl_dir = os.path.join(self.output_dir, f"resolution_level_{res_lvl}")
            os.mkdir(res_lvl_dir)
            dims = self._get_downsampled_dims(volume_dims, res_lvl)
            nbr_chunks = self._get_nbr_chunks(volume_dims, chunk_size, res_lvl)
            # Note: it is crucial that paddding is added per-resolution level (as opposed to being added once at highest
            # resolution level then downsampling)
            pads = self._get_paddings(volume_dims, chunk_size, res_lvl)
            averaged_slices_data: npt.NDArray[np.float32] | None = None
            nbr_written_slices: int = 0

            # read one slice at a time and write the data to the corresponding chunks (and other structures)
            for slice_idx in trange(
                len(self.sorted_image_fps), desc=f"writing chunks for resolution level {res_lvl}", unit="slices"
            ):
                # read and downsample current slice
                if force_8bit_conversion:
                    img = cv.imread(self.sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE)
                else:
                    img = cv.imread(self.sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)

                # write slice data to the residency octree and add its histogram
                if res_lvl == 0:
                    octree_utils.write_slice_to_octree(img, slice_idx, residency_octree, volume_dims)
                    hist += np.histogram(
                        img,
                        bins=histogram_nbr_bins,
                        range=(np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
                        density=False,
                    )[0].astype(np.uint64)

                # downsample slice according to current resolution level
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
                        self._write_slice(
                            data, chunk_size, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir
                        )
                        nbr_written_slices += 1
                    # overwrite/initialize the averaged slices data
                    averaged_slices_data = np.zeros(
                        (nbr_chunks[1] * chunk_size, nbr_chunks[0] * chunk_size),
                        dtype=np.float32,
                    )
                averaged_slices_data += padded_slice_data

            # don't forget to write the last slice!
            data = averaged_slices_data.astype(dtype)
            self._write_slice(data, chunk_size, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir)
            nbr_written_slices += 1

            # finally, write the zero-padding slices
            zeros = np.zeros((nbr_chunks[1] * chunk_size, nbr_chunks[0] * chunk_size), dtype=dtype)
            for _ in range(pads[2]):
                self._write_slice(zeros, chunk_size, nbr_chunks[0], nbr_chunks[1], nbr_written_slices, res_lvl_dir)
                nbr_written_slices += 1
            assert nbr_written_slices == nbr_chunks[2] * chunk_size, (
                f"unexpected number of written slice for the resolution level: {res_lvl}"
            )

        # compute vdhm for different tolerance values
        vdhms = [
            (t, vdhm_penalty, octree_utils.VDHM(residency_octree, node_idx=0, tolerance=t, penalty=vdhm_penalty))
            for t in range(*vdhm_tolerance_range)
        ]

        # set the CVDS metadata
        self.metadata = CVDSMetadata(
            original_dims=imgs_metadata.original_dims.tolist(),
            chunk_size=chunk_size,
            color_depth=imgs_metadata.color_depth,
            force_8bit_conversion=force_8bit_conversion,
            nbr_resolution_lvls=nbr_resolution_levels,
            nbr_chunks_per_resolution_lvl=nbr_chunks_per_res_lvl,
            total_nbr_chunks=total_nbr_chunks,
            lz4_compressed=lz4_compressed,
            decompressed_chunk_size_in_bytes=decompressed_chunk_size_in_bytes,
            downsampling_inter="trilinear",
            octree_max_depth=max_octree_depth,
            octree_nrb_nodes=len(residency_octree),
            octree_size_in_bytes=residency_octree.nbytes,
            octree_smallest_subdivision=(volume_dims * 2**-max_octree_depth).tolist(),
            vdhms=vdhms,
            histogram_nbr_bins=histogram_nbr_bins,
            voxel_dims=(voxel_dim_x, voxel_dim_y, voxel_dim_z),
            euler_rotation=(euler_rot_x, euler_rot_y, euler_rot_z),
        )

        # write the metadata file
        self._write_metadata()

        # perform post processing (e.g., to compress the chunks because stream compression is a bitch)
        self._post_processing()

        # write the recidency octree
        with open(os.path.join(self.output_dir, "residency_octree.bin"), mode="wb") as bs:
            bs.write(residency_octree.tobytes())

        # write the histogram
        with open(os.path.join(self.output_dir, "histogram.bin"), mode="wb") as bs:
            bs.write(hist.tobytes())

    def convert_cvds_dataset(
        self,
        cvds_dataset_dir: str,
        output_dir: str,
        resolution_lvl: int = 0,
        format: str = "PNG",
        visualize_homogeneous_regions: bool = False,
        vdhm_tolerance: int = 1,
    ) -> None:
        """Converts a given CVDS dataset to a set of images. This is mainly used for debugging/testing purposes

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

        residency_octree: npt.NDArray | None = None
        if visualize_homogeneous_regions and resolution_lvl == 0:
            residency_octree = octree_utils.import_octree(cvds_dataset_dir)

        volume_dims = np.asarray(metadata.original_dims, dtype=np.uint32)

        for slice_idx in trange(
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
                chunk_id = i + nbr_chunks_x * nbr_chunks_y * (slice_idx // metadata.chunk_size)
                chunk_fp: str
                chunk_slice: np.memmap
                offset = (slice_idx % metadata.chunk_size) * metadata.chunk_size * metadata.chunk_size * bpp
                if metadata.lz4_compressed:
                    chunk_fp = os.path.join(res_lvl_chunks_path, f"chunk_{chunk_id}.cvds.lz4")
                    decompressed_data: bytes
                    with open(chunk_fp, mode="rb") as bs:
                        decompressed_data = lz4.block.decompress(bs.read(), uncompressed_size=metadata.chunk_size**3)
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

            img_fp: str = os.path.join(output_dir, f"{slice_idx}.{format.lower()}")

            # in case visualizing the octree's homogeneous nodes is intended, draw rectangles
            if visualize_homogeneous_regions and resolution_lvl == 0 and slice_idx < volume_dims[2]:
                # has to be converted to RGB to see the red/green rectangles
                img_data = cv.cvtColor(img_data, cv.COLOR_GRAY2RGB)
                img_data = octree_utils.draw_homogeneous_regions(
                    residency_octree,
                    node_idx=0,
                    slice_data=img_data,
                    slice_idx=slice_idx,
                    vdhm_tolerance=vdhm_tolerance,
                    volume_dims=volume_dims,
                )
            cv.imwrite(img_fp, img_data)

    # TODO: directly use stream compression instead
    def _post_processing(self):
        # do nothing LZ4 compression is disabled
        if not self.metadata.lz4_compressed:
            return
        logging.info("starting post processing step(s) ...")
        for res_lvl in range(0, self.metadata.nbr_resolution_lvls):
            res_lvl_dir = os.path.join(self.output_dir, f"resolution_level_{res_lvl}")
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
        data: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
        chunk_size: int,
        nbr_chunks_x: int,
        nbr_chunks_y: int,
        nbr_written_slices: int,
        resolution_lvl_dir: str,
    ) -> None:
        for i in range(nbr_chunks_x * nbr_chunks_y):
            chunk_id = i + (nbr_written_slices // chunk_size) * nbr_chunks_x * nbr_chunks_y
            fp = os.path.join(resolution_lvl_dir, f"chunk_{chunk_id}.cvds")
            with open(fp, mode="ab") as binary_stream:
                row_start_idx = chunk_size * (i // nbr_chunks_x)
                row_end_idx = row_start_idx + chunk_size
                col_start_idx = chunk_size * (i % nbr_chunks_x)
                col_end_idx = col_start_idx + chunk_size
                binary_stream.write(data[row_start_idx:row_end_idx, col_start_idx:col_end_idx].tobytes())

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
        output_dir=r"output",
        chunk_size=128,
        lz4_compressed=True,
        force_8bit_conversion=True,
        octree_min_nbr_voxels=16,
        vdhm_tolerance_range=(0, 50),
    )
    # converter.convert_cvds_dataset(
    #     r"output",
    #     output_dir="tmp",
    #     resolution_lvl=0,
    #     visualize_homogeneous_regions=True,
    #     vdhm_tolerance=50,
    # )
    print("done")
