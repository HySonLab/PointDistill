"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
# import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define fixed RGB colors (normalized to [0,1]) for Nuscenes
fixed_colors = np.array([
    [0.5, 0.5, 0.5],      # barrier
    [0.0, 0.6, 1.0],      # bicycle
    [1.0, 0.0, 0.0],      # bus
    [1.0, 0.7, 0.2],      # car
    [0.3, 0.3, 0.3],      # construction_vehicle
    [0.6, 0.0, 1.0],      # motorcycle
    [1.0, 0.0, 0.2],      # pedestrian
    [1.0, 0.6, 0.6],      # traffic_cone
    [0.4, 0.2, 0.1],      # trailer
    [0.0, 0.0, 1.0],      # truck
    [0.4, 0.3, 0.2],      # driveable_surface
    [0.6, 0.4, 0.3],      # other_flat
    [1.0, 0.4, 1.0],      # sidewalk
    [1.0, 0.8, 1.0],      # terrain
    [0.7, 0.7, 0.7],      # manmade
    [0.0, 1.0, 0.0],      # vegetation (green!)
])


def add_legend(ax, segment_pred, class_names):
    unique_labels = np.unique(segment_pred)
    seen_labels = set()
    legend_elements = []

    for label in unique_labels:
        if label == -1 or label in seen_labels or label >= len(class_names):
            continue
        seen_labels.add(label)

        class_name = class_names[label]
        color = fixed_colors[label]
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      label=class_name,
                                      markerfacecolor=color, markersize=6))
    ax.legend(handles=legend_elements, title="Classes",
              loc='upper right', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)


def get_label_colors(labels):
    # Set default color (e.g., light gray) for ignored labels
    default_color = np.array([0.8, 0.8, 0.8])  # light gray
    colors = np.array([default_color] * len(labels))

    valid_mask = labels != -1
    colors[valid_mask] = fixed_colors[labels[valid_mask]]
    return colors


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


# # Function to map class labels to colors
# def get_label_colors(labels, num_classes):
#     # Use Matplotlib's tab20 colormap for distinct colors
#     cmap = plt.get_cmap("tab20")
#     colors = np.array([cmap(i % 20)[:3] for i in range(num_classes)])  # RGB colors
#     # Map labels to colors
#     label_colors = colors[labels % num_classes]  # Handle any out-of-range labels
#     return label_colors


# def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     coord = to_numpy(coord)
#     if color is not None:
#         color = to_numpy(color)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coord)
#     pcd.colors = o3d.utility.Vector3dVector(
#         np.ones_like(coord) if color is None else color
#     )
#     o3d.io.write_point_cloud(file_path, pcd)
#     if logger is not None:
#         logger.info(f"Save Point Cloud to: {file_path}")


# def save_bounding_boxes(
#     bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
# ):
#     bboxes_corners = to_numpy(bboxes_corners)
#     # point list
#     points = bboxes_corners.reshape(-1, 3)
#     # line list
#     box_lines = np.array(
#         [
#             [0, 1],
#             [1, 2],
#             [2, 3],
#             [3, 0],
#             [4, 5],
#             [5, 6],
#             [6, 7],
#             [7, 0],
#             [0, 4],
#             [1, 5],
#             [2, 6],
#             [3, 7],
#         ]
#     )
#     lines = []
#     for i, _ in enumerate(bboxes_corners):
#         lines.append(box_lines + i * 8)
#     lines = np.concatenate(lines)
#     # color list
#     color = np.array([color for _ in range(len(lines))])
#     # generate line set
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(color)
#     o3d.io.write_line_set(file_path, line_set)
#
#     if logger is not None:
#         logger.info(f"Save Boxes to: {file_path}")
#
#
# def save_lines(
#     points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
# ):
#     points = to_numpy(points)
#     lines = to_numpy(lines)
#     colors = np.array([color for _ in range(len(lines))])
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_line_set(file_path, line_set)
#
#     if logger is not None:
#         logger.info(f"Save Lines to: {file_path}")
