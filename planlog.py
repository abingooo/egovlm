# 图像对象检测类 - 用于检测图像中的对象
from google import genai
from google.genai import types
import cv2
import numpy as np
import json
import time
from lib.vlm import ImageObjectDetector
from lib.logread import create_reader
from lib.draw_utils import DrawUtils, draw_and_save
from lib.solve_utils import calculate_depth_from_detections, calculate_3d_position
from lib.cloud_utils import PointCloudUtils
# 使用示例
if __name__ == "__main__":
    # 创建日志读取器
    reader = create_reader('./log')
    
    # 读取最新的日志数据
    latest_data = reader.read_latest_log()
    # 获取RGB图像
    rgb_image = latest_data['color_image']
    # 获取深度数据
    dep_data = latest_data['depth_data']
    # 创建对象检测器，指定检测类型为box
    detector = ImageObjectDetector(detection_type="waypoint1")
    start_time = time.time()
    # 检测对象（设置verbose=False避免重复打印）
    detections = detector.detect_objects(
        rgb_image=rgb_image,
        # control_instruction="fly to the chair on the left(3.109m, 2.068m, -0.835m)",
        # control_instruction="fly to the chair on the left",
        # control_instruction="fly to the water dispenser(8.475m,-1.771m, -0.094m)",
        control_instruction="Fly to the water dispenser(8.475m,-1.771m, -0.094m)  and then back to 1 meter above the original position",
        verbose=False
    )
    print(f"推理时间: {time.time() - start_time:.2f} 秒")
    print(detections)

    if detector.detection_type == "waypoint1":
        point_cloud_utils = PointCloudUtils()
        annotation_points = point_cloud_utils.convert_labeled_points(detections)
        # 生成递增序号的输出文件名，避免覆盖
        input_ply_path = "./log/20251025_221217/point_cloud.ply"
        output_ply_path = "./log/20251025_221217/point_cloud_waypoint.ply"
        print(f"将生成带标注的点云文件: {output_ply_path}")
        result = point_cloud_utils.add_annotation_points(input_ply_path, output_ply_path, annotation_points, radius=0.1, points_per_arrow=4000)
        print(f"点云标注结果: {result}")
    else:

        # 使用便捷函数绘制并显示结果
        draw_and_save(rgb_image, detections, detection_type=detector.detection_type, save_path="./detection_result.jpg")

        # 使用solve_utils中的函数计算所有检测对象的深度值
        depth_values = calculate_depth_from_detections(detections, dep_data)
        
        
        # 计算3D位置
        positions_3d = calculate_3d_position(detections, dep_data, verbose=False, convert_coordinate_system=True)
        
        # 打印每个检测对象的深度值和3D坐标
        for i, detection in enumerate(detections):
            if i < len(depth_values) and depth_values[i] is not None:
                if "point" in detection:
                    y, x = detection["point"]
                    print(f"对象:{detection['label']} 像素坐标: ({y}, {x})")
                    print(f"  深度值: {depth_values[i]:.3f}")
                elif "box" in detection:
                    ymin, xmin, ymax, xmax = detection['box']
                    print(f"对象:{detection['label']} 边界框: [{ymin}, {xmin}, {ymax}, {xmax}]")
                    print(f"  深度值: {depth_values[i]*0.001:.3f} 米")
                
                # 打印3D坐标
                if i < len(positions_3d) and positions_3d[i] is not None:
                    X, Y, Z = positions_3d[i]
                    print(f"  3D坐标: {X:.3f}m, {Y:.3f}m, {Z:.3f}米m")
            elif i < len(depth_values) and depth_values[i] is None:
                print(f"对象:{detection['label']} 无法获取深度值")
                if i < len(positions_3d) and positions_3d[i] is None:
                    print(f"  无法计算3D坐标")
