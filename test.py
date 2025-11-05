#!/usr/bin/env python3
"""
Visualize 3D bounding box points (front/back) and center.
- Uses matplotlib (no seaborn).
- Shows 9 points and box edges with equal aspect.
- Prints size W/H/D in the title.

Run:
    python visualize_bbox_points.py
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

def visualize_bbox(item):
    center = item["center"]
    front = item["bbox3dfront"]
    back = item["bbox3dback"]

    # Compute dimensions
    xs = [p[0] for p in front + back]
    ys = [p[1] for p in front + back]
    # Front/Back share the same z per face; use first points for depth
    depth_z = back[0][2] - front[0][2]
    width_x = max(xs) - min(xs)
    height_y = max(ys) - min(ys)

    # Prepare arrays
    front_xyz = np.array(front, dtype=float)
    back_xyz  = np.array(back, dtype=float)
    center_np = np.array(center, dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Points (no explicit colors/styles)
    ax.scatter(front_xyz[:,0], front_xyz[:,1], front_xyz[:,2], marker='o', s=40, label='front pts')
    ax.scatter(back_xyz[:,0], back_xyz[:,1], back_xyz[:,2], marker='s', s=40, label='back pts')
    ax.scatter([center_np[0]], [center_np[1]], [center_np[2]], marker='^', s=100, label='center')

    # Edges for front/back (perimeter order)
    edge_order = [(0,2), (2,1), (1,3), (3,0)]
    for a, b in edge_order:
        ax.plot([front[a][0], front[b][0]], [front[a][1], front[b][1]], [front[a][2], front[b][2]])
        ax.plot([back[a][0],  back[b][0]],  [back[a][1],  back[b][1]],  [back[a][2],  back[b][2]])

    # Connect corresponding vertices
    for i in range(4):
        ax.plot([front[i][0], back[i][0]], [front[i][1], back[i][1]], [front[i][2], back[i][2]])

    # Labels
    for i, p in enumerate(front):
        ax.text(p[0], p[1], p[2], f"F{i}", fontsize=9)
    for i, p in enumerate(back):
        ax.text(p[0], p[1], p[2], f"B{i}", fontsize=9)
    ax.text(center_np[0], center_np[1], center_np[2], "C", fontsize=10)

    # Equal aspect box
    all_pts = np.vstack([front_xyz, back_xyz, center_np[None, :]])
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    ranges = xyz_max - xyz_min
    max_range = ranges.max()
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D BBox for '{item.get('label', 'object')}' | "
        f"size: W={width_x:.3f}, H={height_y:.3f}, D={depth_z:.3f} | "
        f"center=({center_np[0]:.3f}, {center_np[1]:.3f}, {center_np[2]:.3f})"
    )
    ax.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    # ---- Paste your data here ----
    item = {
        "id": 0,
        "label": "tree",
        "depth": 3691.39990234375,
        "center": [0.9300000071525574, -0.2199999988079071, 4.642205715179443],
        "bbox3dfront": [
            [-1.2000000476837158, -1.2899999618530273, 3.690000057220459],
            [ 2.369999885559082 ,  1.8600000143051147, 3.690000057220459],
            [ 2.369999885559082 , -1.2899999618530273, 3.690000057220459],
            [-1.2000000476837158,  1.8600000143051147, 3.690000057220459],
        ],
        "bbox3dback": [
            [-1.2000000476837158, -1.2899999618530273, 5.594411849975586],
            [ 2.369999885559082 ,  1.8600000143051147, 5.594411849975586],
            [ 2.369999885559082 , -1.2899999618530273, 5.594411849975586],
            [-1.2000000476837158,  1.8600000143051147, 5.594411849975586],
        ],
    }

    visualize_bbox(item)
