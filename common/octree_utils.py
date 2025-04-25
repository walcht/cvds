import numpy as np
import numpy.typing as npt
import os
import cv2 as cv
from cv2.typing import MatLike

octree_node_dtype = np.dtype(
    [
        ("center_x", np.float32),  # 4 bytes
        ("center_y", np.float32),  # 4 bytes
        ("center_z", np.float32),  # 4 bytes
        ("side_halved", np.float32),  # 4 bytes
        ("data", np.uint32),  # 4 bytes
    ]
)


def create_and_initialize_octree(
    volume_dims: npt.NDArray[np.uint32], min_nbr_voxels: int = 32
) -> tuple[npt.NDArray, int]:
    """Creates and initializes a residency octree data structure

    Parameters
    ----------
    volume_dims : npt.NDArray[np.uint32]
        original dimensions of the visualized volume. Used to determine voxel dimensions in normalized volume space.

    min_nbr_voxels : int, optional
        minimum number of voxels along each octree node's dimensions the maximum depth has to satisfy, by default 32

    Returns
    -------
    tuple[npt.NDArray, int]
        First field is the initialized residency octree data structure with mins and maxes set to 255 and 0,
        respectively. Second field is the max depth of the residency octree (inclusive).
    """

    # use a heuristic to determine max depth
    max_octree_depth: int = get_max_octree_depth(volume_dims, min_nbr_voxels)

    octree_nbr_elements: int = (int)((8 ** (max_octree_depth + 1) - 1) / 7)

    residency_octree = np.zeros(
        octree_nbr_elements,
        dtype=octree_node_dtype,
    )

    # initialize first node
    residency_octree[0] = (0.5, 0.5, 0.5, 0.5, 0x000000FF)

    # recursively initialize the remaining nodes
    populate_children(residency_octree, 0)

    return residency_octree, max_octree_depth


def get_max_octree_depth(volume_dims: npt.NDArray[np.uint32], min_nbr_voxels: int = 32) -> int:
    """Uses a heuristic to determine the maximum octree depth.

    Since octrees are used for empty space skipping, skipping very small regions may significantly hurt the performance.
    It is, therefore, beneficial to limit the octree traversal depth to, say, a point where it no longer makes sense,
    performance wise, to skip any smaller regions. The heuristic used to determine this maximum depth is to determine
    the depth such that at least one of the octree node's dimensions is no longer greater than or equal to
    *min_nbr_voxels*, the max depth is then set to the depth above it, if any.

    Parameters
    ----------
    volume_dims : npt.NDArray[np.uint32]
        original dimensions of the visualized volume. Used to determine voxel dimensions in normalized volume space.

    min_nbr_voxels : int, optional
        minimum number of voxels along each octree node's dimensions the maximum depth has to satisfy, by default 32

    Returns
    -------
    int
        inclusive max octree depth
    """

    voxel_dims = 1 / volume_dims
    d = -np.log2(min_nbr_voxels * voxel_dims)
    return np.floor(np.min(d))


def populate_children(octree: npt.NDArray, parent_node_idx: int) -> None:
    """Recursively initializes the children of the provided residency octree node

    Parameters
    ----------
    residency_octree : npt.NDArray
        residency octree to be initialized. It is assumed that root node is already initialized

    parent_node_idx : int
        index of the parent node whose children are going to be initialized
    """

    if (8 * parent_node_idx + 1) >= len(octree):
        return
    side_halved = octree[parent_node_idx]["side_halved"] / 2
    data = 0x000000FF

    # child 1
    child_idx = 8 * parent_node_idx + 1
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] - side_halved,
        octree[parent_node_idx]["center_y"] - side_halved,
        octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 2
    child_idx = 8 * parent_node_idx + 2
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] + side_halved,
        octree[parent_node_idx]["center_y"] - side_halved,
        octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 3
    child_idx = 8 * parent_node_idx + 3
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] - side_halved,
        octree[parent_node_idx]["center_y"] + side_halved,
        octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 4
    child_idx = 8 * parent_node_idx + 4
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] + side_halved,
        octree[parent_node_idx]["center_y"] + side_halved,
        octree[parent_node_idx]["center_z"] - side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 5
    child_idx = 8 * parent_node_idx + 5
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] - side_halved,
        octree[parent_node_idx]["center_y"] - side_halved,
        octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 6
    child_idx = 8 * parent_node_idx + 6
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] + side_halved,
        octree[parent_node_idx]["center_y"] - side_halved,
        octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 7
    child_idx = 8 * parent_node_idx + 7
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] - side_halved,
        octree[parent_node_idx]["center_y"] + side_halved,
        octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)

    # child 8
    child_idx = 8 * parent_node_idx + 8
    octree[child_idx] = (
        octree[parent_node_idx]["center_x"] + side_halved,
        octree[parent_node_idx]["center_y"] + side_halved,
        octree[parent_node_idx]["center_z"] + side_halved,
        side_halved,
        data,
    )
    populate_children(octree, child_idx)


def write_slice_to_octree(
    slice_data: MatLike,
    slice_idx: int,
    octree: npt.NDArray,
    volume_dims: npt.NDArray[np.uint32],
) -> None:
    """_summary_

    Parameters
    ----------
    slice_data : MatLike
        _description_
    slice_idx : int
        _description_
    octree : npt.NDArray
        _description_
    volume_dims : npt.NDArray[np.uint32]
        _description_
    """

    voxel_spatial_extent = 1.0 / volume_dims

    for i in range(len(octree)):
        slice_pos_z = (voxel_spatial_extent[2] / 2.0) + slice_idx * voxel_spatial_extent[2]
        side_halved = octree[i]["side_halved"]
        if (slice_pos_z > (octree[i]["center_z"] + side_halved)) or (
            slice_pos_z < (octree[i]["center_z"] - side_halved)
        ):
            # slice does not contribute to the node's min/max
            continue

        # retrieve subregion of the image slice that contributes to the node's min/max
        y_start_idx = (int)((octree[i]["center_y"] - side_halved) / voxel_spatial_extent[1])
        y_end_idx = (int)((octree[i]["center_y"] + side_halved) / voxel_spatial_extent[1])

        x_start_idx = (int)((octree[i]["center_x"] - side_halved) / voxel_spatial_extent[0])
        x_end_idx = (int)((octree[i]["center_x"] + side_halved) / voxel_spatial_extent[0])

        sub_slice = slice_data[y_start_idx : y_end_idx + 1, x_start_idx : x_end_idx + 1]

        node_min = (octree[i]["data"] >> 0) & 0xFF
        node_min = min(node_min, sub_slice.min())

        node_max = (octree[i]["data"] >> 8) & 0xFF
        node_max = max(node_max, sub_slice.max())

        # write back the potentially new min/max node values
        octree[i]["data"] = (octree[i]["data"] & 0xFFFF0000) | (np.uint32(node_max) << 8) | (node_min << 0)


def VDHM(residency_octree: npt.NDArray, node_idx: int, tolerance: int = 0, penalty: float = 8.05) -> float:
    # in case this is a homogeneous node, return 1
    node_min = (residency_octree[node_idx]["data"] >> 0) & 0xFF
    node_max = (residency_octree[node_idx]["data"] >> 8) & 0xFF
    if (node_max - node_min) <= tolerance:
        return 1

    # in case this is a non-homogeneous leaf node, return 0
    if (8 * node_idx + 1) >= len(residency_octree):
        return 0

    # otherwise recursively compute the VDHM of the suboctree
    accm: float = 0.0
    for i in range(1, 9):
        child_idx = 8 * node_idx + i
        accm += VDHM(residency_octree, child_idx, tolerance, penalty)

    return accm / penalty


def draw_homogeneous_regions(
    residency_octree: npt.NDArray,
    node_idx: int,
    slice_data: npt.NDArray,
    slice_idx: int,
    vdhm_tolerance: int,
    volume_dims: npt.NDArray,
) -> None:
    node = residency_octree[node_idx]
    # in case this node falls out of the spatial extent of the provided slice, do nothing
    voxel_size = 1.0 / volume_dims
    slice_normalized_z_coord = slice_idx * voxel_size[2]
    if (slice_normalized_z_coord < (node["center_z"] - node["side_halved"])) or (
        slice_normalized_z_coord >= (node["center_z"] + node["side_halved"])
    ):
        return slice_data

    # in case this is a homogeneous node, draw green rectangle and return the modified slice data
    node_min = (node["data"] >> 0) & 0xFF
    node_max = (node["data"] >> 8) & 0xFF
    if (node_max - node_min) <= vdhm_tolerance:
        slice_data = cv.rectangle(
            slice_data,
            (
                round((node["center_x"] - node["side_halved"]) * volume_dims[0]),
                round((node["center_y"] - node["side_halved"]) * volume_dims[1]),
            ),
            (
                round((node["center_x"] + node["side_halved"]) * volume_dims[0]),
                round((node["center_y"] + node["side_halved"]) * volume_dims[1]),
            ),
            (0, 255, 0),
            1,
        )
        return slice_data

    # in case this is a non-homogeneous leaf node, dont draw anything
    if (8 * node_idx + 1) >= len(residency_octree):
        return slice_data

    # otherwise, recursively check child nodes and draw rectangles if necessary
    for i in range(1, 9):
        child_idx = 8 * node_idx + i
        slice_data = draw_homogeneous_regions(
            residency_octree,
            child_idx,
            slice_data,
            slice_idx,
            vdhm_tolerance,
            volume_dims,
        )

    return slice_data


def import_octree(cvds_dir_path: str) -> npt.NDArray:
    return np.fromfile(os.path.join(cvds_dir_path, "residency_octree.bin"), dtype=octree_node_dtype)
