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
    detector = ImageObjectDetector(detection_type="flypoint")
    start_time = time.time()
    # 检测对象（设置verbose=False避免重复打印）
    detections = detector.detect_objects(
        rgb_image=rgb_image,
        control_instruction=["fly to the water dispenser","fly to the bed"],
        verbose=False
    )
    print(f"推理时间: {time.time() - start_time:.2f} 秒")
    print(detections)

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
                print(f"  3D坐标: X={X:.3f}米, Y={Y:.3f}米, Z={Z:.3f}米")
        elif i < len(depth_values) and depth_values[i] is None:
            print(f"对象:{detection['label']} 无法获取深度值")
            if i < len(positions_3d) and positions_3d[i] is None:
                print(f"  无法计算3D坐标")
