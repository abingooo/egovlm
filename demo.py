import open3d as o3d
import numpy as np

ply_path = "./log/point_cloud.ply" 
pcd = o3d.io.read_point_cloud(ply_path)
print(pcd)
o3d.visualization.draw_geometries([pcd], window_name="Project point cloud")