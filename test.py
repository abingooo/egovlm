from lib.cloud_utils import PointCloudUtils
import json

with open('./points.json', 'r') as f:
        path_points = json.load(f)
        
pcu = PointCloudUtils()
pcu.process_point_cloud(
                        input_ply_path="./log/point_cloud_cube.ply",
                        annotation_data=path_points, 
                        modeling_type="path", 
                        output_ply_path="./log/point_cloud_cube_path.ply", 
                        show_visualization=True, 
                        radius=0.1
                        )
    