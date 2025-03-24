import numpy as np
import cv2 as cv
from tqdm import trange
import glob
import os


def get_sorted_images_from_dir(
    dataset_dir: str,
    verbose: bool = True,
) -> list[str]:
    supported_formats: dict[str, str] = {"tiff": ".tif", "png": ".png"}
    for supported_format, ext in supported_formats.items():
        fps = [fp for fp in glob.glob(os.path.join(dataset_dir, f"*{ext}"), recursive=False)]
        if fps:
            if verbose:
                print(f"found: {len(fps)} {supported_format.upper()} image files")
            # order according to filename - very important!
            return sorted(fps)
    if verbose:
        print("could not find any supported image files")
    return []


def get_metadata(
    sorted_image_fps: list[str],
    brick_size: int,
    max_octree_depth: int,
):
    img = cv.imread(sorted_image_fps[0], cv.IMREAD_GRAYSCALE)
    original_dims = np.array([img.shape[1], img.shape[0], len(sorted_image_fps)], dtype=np.uint32)
    dims = np.ceil(original_dims / brick_size).astype(np.uint32) * brick_size
    return {
        "brick_size": brick_size,
        "max_octree_depth": max_octree_depth,
        "original_dims": original_dims,
        "dims": dims,
    }


def generate_residency_octree(
    sorted_image_fps: list[str],
    metadata: dict,
) -> np.ndarray:
    dims = metadata["dims"]
    original_dims = metadata["original_dims"]
    max_octree_depth = metadata["max_octree_depth"]
    nbr_elements = (int)((8 ** (max_octree_depth + 1) - 1) / 7)
    residency_octree = np.zeros(
        nbr_elements,
        dtype=[
            ("center_x", np.float32),  # 4 bytes
            ("center_y", np.float32),  # 4 bytes
            ("center_z", np.float32),  # 4 bytes
            ("side_halved", np.float32),  # 4 bytes
            ("data", np.uint32),  # 4 bytes
        ],
    )

    # log some useful info
    print(f"number of nodes: {nbr_elements}")

    # fill first node data
    residency_octree[0] = (0.5, 0.5, 0.5, 0.5, 0x000000FF)
    populate_children(residency_octree, 0)

    voxel_spatial_extent = (1.0 / dims[0], 1.0 / dims[1], 1.0 / dims[2])
    for slice_idx in trange(len(sorted_image_fps), desc="reading image slice", unit="slice"):
        img = np.pad(
            cv.imread(sorted_image_fps[slice_idx], cv.IMREAD_GRAYSCALE),
            pad_width=((0, dims[1] - original_dims[1]), (0, dims[0] - original_dims[0])),
            constant_values=(0,),
        )
        for i in range(len(residency_octree)):
            slice_pos_z = (voxel_spatial_extent[2] / 2.0) + slice_idx * voxel_spatial_extent[2]
            side_halved = residency_octree[i]["side_halved"]
            if (slice_pos_z > (residency_octree[i]["center_z"] + side_halved)) or (
                slice_pos_z < (residency_octree[i]["center_z"] - side_halved)
            ):
                # slice does not contribute to the node's min/max
                continue
            # retrieve subregion of the image slice that contributes to the node's min/max
            y_start_idx = (int)((residency_octree[i]["center_y"] - side_halved) / voxel_spatial_extent[1])
            y_end_idx = (int)((residency_octree[i]["center_y"] + side_halved) / voxel_spatial_extent[1])
            x_start_idx = (int)((residency_octree[i]["center_x"] - side_halved) / voxel_spatial_extent[0])
            x_end_idx = (int)((residency_octree[i]["center_x"] + side_halved) / voxel_spatial_extent[0])
            sub_slice = img[y_start_idx : y_end_idx + 1, x_start_idx : x_end_idx + 1]
            node_min = (residency_octree[i]["data"] >> 0) & 0xFF
            node_min = min(node_min, sub_slice.min())
            node_max = (residency_octree[i]["data"] >> 8) & 0xFF
            node_max = max(node_max, sub_slice.max())
            # finally write back the potentially new min/max node values
            residency_octree[i]["data"] = (
                (residency_octree[i]["data"] & 0xFFFF0000) | (np.uint32(node_max) << 8) | (node_min << 0)
            )
    return residency_octree


def populate_children(residency_octree: np.ndarray, parent_node_idx: int) -> None:
    """Recursively initializes the children of the provided residency octree node

    Parameters
    ----------
    residency_octree : np.ndarray
        residency octree to be initialized. It is assumed that root node is already
        initialized
    parent_node_idx : int
        index of the parent node whose children are going to be initialized
    """
    if (8 * parent_node_idx + 1) >= len(residency_octree):
        return
    side_halved = residency_octree[parent_node_idx]["side_halved"] / 2
    data = 0x000000FF
    # for i in range(1, 9):

    # child 1
    child_idx = 8 * parent_node_idx + 1
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] - side_halved,
        residency_octree[parent_node_idx]["center_y"] - side_halved,
        residency_octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 2
    child_idx = 8 * parent_node_idx + 2
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] + side_halved,
        residency_octree[parent_node_idx]["center_y"] - side_halved,
        residency_octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 3
    child_idx = 8 * parent_node_idx + 3
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] - side_halved,
        residency_octree[parent_node_idx]["center_y"] + side_halved,
        residency_octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 4
    child_idx = 8 * parent_node_idx + 4
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] + side_halved,
        residency_octree[parent_node_idx]["center_y"] + side_halved,
        residency_octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 5
    child_idx = 8 * parent_node_idx + 5
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] - side_halved,
        residency_octree[parent_node_idx]["center_y"] - side_halved,
        residency_octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 6
    child_idx = 8 * parent_node_idx + 6
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] + side_halved,
        residency_octree[parent_node_idx]["center_y"] - side_halved,
        residency_octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 7
    child_idx = 8 * parent_node_idx + 7
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] - side_halved,
        residency_octree[parent_node_idx]["center_y"] + side_halved,
        residency_octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)

    # child 8
    child_idx = 8 * parent_node_idx + 8
    residency_octree[child_idx] = (
        residency_octree[parent_node_idx]["center_x"] + side_halved,
        residency_octree[parent_node_idx]["center_y"] + side_halved,
        residency_octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(residency_octree, child_idx)


def get_homogeneous_children_nodes(
    residency_octree: np.ndarray,
    parent_node_idx: int,
) -> list[int]:
    if (8 * parent_node_idx + 1) >= len(residency_octree):
        return []
    res: list[int] = []
    for i in range(1, 9):
        child_idx = 8 * parent_node_idx + i
        if ((residency_octree[child_idx]["data"] >> 0) & 0xFF) == ((residency_octree[child_idx]["data"] >> 8) & 0xFF):
            res.append(child_idx)
        else:
            res.extend(get_homogeneous_children_nodes(residency_octree, child_idx))
    return res


def compute_stats(
    residency_octree: np.ndarray,
    metadata: dict,
) -> dict:
    """Computes some statistics for the provided residency octree

    Parameters
    ----------
    residency_octree : np.ndarray
        the residency octree out of which stats are computed
    metadata : dict
        additional metadata file containing dimension information, brick size, etc.

    Returns
    -------
    dict
        returns stats dictionary
    """
    res = {
        "nbr_homogeneous_nodes": 0,
        "skippable_portion": 0.0,
        "nbr_homogeneous_nodes_per_depth_lvl": [0 for i in range(metadata["max_octree_depth"] + 1)],
        "smallest_subdivision": metadata["dims"] * 2 ** -metadata["max_octree_depth"],
        "size_in_bytes": residency_octree.nbytes,
        "nbr_nodes": len(residency_octree),
    }
    skippable_nodes: list[int]
    if ((residency_octree[0]["data"] >> 0) & 0xFF) == ((residency_octree[0]["data"] >> 8) & 0xFF):
        skippable_nodes = [0]
    else:
        skippable_nodes = get_homogeneous_children_nodes(residency_octree, 0)

    for skippable_node_idx in skippable_nodes:
        res["skippable_portion"] += residency_octree[skippable_node_idx]["side_halved"] ** 3 / 1.0
        depth_lvl = (int)(- (1 + np.log2(residency_octree[skippable_node_idx]["side_halved"])))
        res["nbr_homogeneous_nodes_per_depth_lvl"][depth_lvl] += 1
    res["skippable_portion"] *= 100
    res["nbr_homogeneous_nodes"] = len(skippable_nodes)
    return res


def serialize_residency_octree(residency_octree: np.ndarray, output: str) -> None:
    """Serializes the provided residency octree

    Parameters
    ----------
    residency_octree : np.ndarray
        residency octree to serialize
    output : str
        filepath at which the raw bytes of the residency octree will be written
    """
    with open(output, mode="wb") as bs:
        bs.write(residency_octree.tobytes())


if __name__ == "__main__":
    sorted_images_fps = get_sorted_images_from_dir(r"C:\Users\walid\Desktop\CTDatasets\dataset_02")
    metadata = get_metadata(sorted_images_fps, brick_size=128, max_octree_depth=5)
    residency_octree = generate_residency_octree(sorted_images_fps, metadata)
    serialize_residency_octree(residency_octree, "residency_octree.bin")
    stats = compute_stats(residency_octree, metadata)
    print(f"residency octree stats: {stats}")
    print("done")
