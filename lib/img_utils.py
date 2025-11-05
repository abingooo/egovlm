import cv2
import os
import numpy as np
from PIL import Image

class ImageUtils:
    """图像工具类，提供图像保存和绘制功能"""
    
    @staticmethod
    def ensure_directory_exists(directory):
        """确保目录存在，如果不存在则创建
        
        Args:
            directory (str): 目录路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    @staticmethod
    def save_image(image, save_path):
        """保存图像到指定路径
        
        Args:
            image (numpy.ndarray): 要保存的图像
            save_path (str): 保存路径
        
        Returns:
            bool: 保存是否成功
        """
        # 确保保存目录存在
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 保存图像
        success = cv2.imwrite(save_path, image)
        if success:
            print(f"图像已保存到: {save_path}")
        else:
            print(f"图像保存失败: {save_path}")
        
        return success
    
    @staticmethod
    def draw_bounding_boxes(image, detections, box_color=(0, 0, 255), text_color=(0, 255, 0), 
                           font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2):
        """在图像上绘制边界框和标签
        
        Args:
            image (numpy.ndarray): 原始图像
            detections (list): 检测结果列表，每个元素为包含'box_2d'和'label'的字典
            box_color (tuple): 边界框颜色 (BGR格式)
            text_color (tuple): 文本颜色 (BGR格式)
            font: 字体类型
            font_scale: 字体缩放因子
            thickness: 线条粗细
        
        Returns:
            numpy.ndarray: 绘制了边界框的图像
        """
        # 创建图像副本以避免修改原始图像
        image_with_boxes = image.copy()
        
        # 绘制每个检测结果
        for i, detection in enumerate(detections):
            if "box_2d" in detection:
                # 获取边界框坐标 [y1, x1, y2, x2]
                y1, x1, y2, x2 = detection["box_2d"]
                
                # 绘制矩形框
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), box_color, thickness)
                
                # 获取标签
                label = detection.get("label", f"object_{i+1}")
                
                # 计算文本大小以正确定位
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # 绘制文本背景
                cv2.rectangle(image_with_boxes, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width + 5, y1), 
                             box_color, 
                             -1)  # -1 表示填充矩形
                
                # 绘制文本
                cv2.putText(image_with_boxes, label, (x1 + 3, y1 - 5), font, font_scale, text_color, thickness)
        
        return image_with_boxes
    
    @staticmethod
    def save_detection_results(rgb_image, detections, save_dir='./debug', 
                             image_filename='rgb_image_preview.jpg', 
                             result_filename='detection_result_with_boxes.jpg'):
        """保存原始图像和带有检测结果的图像
        
        Args:
            rgb_image (numpy.ndarray): RGB图像
            detections (list): 检测结果列表
            save_dir (str): 保存目录
            image_filename (str): 原始图像文件名
            result_filename (str): 检测结果图像文件名
        
        Returns:
            tuple: (原始图像路径, 结果图像路径)
        """
        # 确保保存目录存在
        ImageUtils.ensure_directory_exists(save_dir)
        
        # 保存原始RGB图像
        image_path = os.path.join(save_dir, image_filename)
        ImageUtils.save_image(rgb_image, image_path)
        
        # 绘制检测结果
        image_with_boxes = ImageUtils.draw_bounding_boxes(rgb_image, detections)
        
        # 保存绘制后的图像
        result_path = os.path.join(save_dir, result_filename)
        ImageUtils.save_image(image_with_boxes, result_path)
        
        return image_path, result_path
    
    @staticmethod
    def convert_lsam_coordinates_to_rgb(lsam_result, roi_bbox):
        """
        将LSAM分割结果的坐标从ROI图像坐标系转换到原始RGB图像坐标系
        
        Args:
            lsam_result (dict): LSam分割结果，包含mask_count、masks等信息
            roi_bbox (list): ROI在RGB图像中的边界框，格式为[[x1, y1], [x2, y2]]
            
        Returns:
            dict: 转换后的LSAM分割结果，其中所有坐标都已转换到RGB图像坐标系
        """
        # 创建结果副本以避免修改原始数据
        converted_result = lsam_result.copy()
        
        # 获取ROI在RGB图像中的偏移量
        x_offset, y_offset = roi_bbox[0]
        
        # 如果有掩码结果，进行坐标转换
        if "mask_count" in converted_result and converted_result["mask_count"] > 0:
            # 遍历每个掩码
            for mask_data in converted_result.get("masks", []):
                # 转换边界框坐标
                if "bounding_box" in mask_data:
                    bbox = mask_data["bounding_box"]
                    bbox["x1"] = bbox.get("x1", 0) + x_offset
                    bbox["x2"] = bbox.get("x2", 0) + x_offset
                    bbox["y1"] = bbox.get("y1", 0) + y_offset
                    bbox["y2"] = bbox.get("y2", 0) + y_offset
                
                # 转换质心坐标
                if "centroid" in mask_data:
                    centroid = mask_data["centroid"]
                    mask_data["centroid"] = [centroid[0] + x_offset, centroid[1] + y_offset]
                
                # 转换随机点坐标
                if "random_points" in mask_data:
                    converted_points = []
                    for point in mask_data["random_points"]:
                        converted_points.append([point[0] + x_offset, point[1] + y_offset])
                    mask_data["random_points"] = converted_points
        
        return converted_result
        
    @staticmethod
    def visualize_lsam_result(roi_image, lsam_result, index=0, label="object", save_dir="./debug/rois"):
        """
        可视化LSAM分割结果并保存到rois目录
        
        Args:
            roi_image (numpy.ndarray): ROI区域图像（RGB格式）
            lsam_result (dict): LSam分割结果
            index (int): 对象索引，用于文件名
            label (str): 对象标签，用于文件名
            save_dir (str): 保存目录
            
        Returns:
            str: 保存的图像路径，如果失败则返回None
        """
        try:
            # 确保保存目录存在
            ImageUtils.ensure_directory_exists(save_dir)
            
            # 创建图像副本以避免修改原始图像
            visualized = roi_image.copy()
            
            # 设置颜色（BGR格式）
            centroid_color = (0, 0, 255)  # 红色 - 质心
            box_color = (0, 255, 0)       # 绿色 - 检测框
            random_color = (255, 165, 0)  # 橙色 - 随机点
            
            # 检查是否有掩码结果
            if "mask_count" in lsam_result and lsam_result["mask_count"] > 0:
                # 遍历每个掩码并绘制
                for mask_data in lsam_result.get("masks", []):
                    # 绘制边界框
                    if "bounding_box" in mask_data:
                        bbox = mask_data["bounding_box"]
                        # 确保边界框坐标完整
                        if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                            cv2.rectangle(visualized, 
                                        (bbox["x1"], bbox["y1"]), 
                                        (bbox["x2"], bbox["y2"]), 
                                        box_color, 2)
                    
                    # 绘制质心
                    if "centroid" in mask_data:
                        centroid = mask_data["centroid"]
                        cv2.circle(visualized, (int(centroid[0]), int(centroid[1])), 5, centroid_color, -1)
                        cv2.putText(visualized, 'Centroid', (int(centroid[0]) + 10, int(centroid[1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, centroid_color, 1)
                    
                    # 绘制随机点
                    if "random_points" in mask_data:
                        for idx, point in enumerate(mask_data["random_points"]):
                            cv2.circle(visualized, (int(point[0]), int(point[1])), 4, random_color, -1)
                            cv2.putText(visualized, f'P{idx+1}', (int(point[0]) + 8, int(point[1]) + 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, random_color, 1)
            else:
                # 如果没有检测结果，添加文本提示
                cv2.putText(visualized, "未检测到对象", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("未检测到对象，返回原图")
            
            # 构建保存路径
            label_safe = label.replace(' ', '_')
            save_path = os.path.join(save_dir, f"lsam_visualization_{index}_{label_safe}.jpg")
            
            # 保存图像
            success = ImageUtils.save_image(visualized, save_path)
            
            if success:
                print(f"LSAM分割结果已可视化并保存到: {save_path}")
                return save_path
            else:
                return None
                
        except Exception as e:
                    print(f"LSAM分割结果可视化过程中发生错误: {str(e)}")
                    return None
    
    @staticmethod
    def visualize_target3d_results(rgb_image, target3d, save_dir='debug', filename='target3d_visualization.jpg'):
        """
        可视化显示target3d结果到RGB图像上
        
        Args:
            rgb_image (numpy.ndarray): RGB图像
            target3d (list): 3D目标列表，每个元素为包含id、label、center、bbox、npoints等信息的字典
            save_dir (str): 保存目录
            filename (str): 保存的文件名
            
        Returns:
            str: 保存的图像路径
        """
        # 创建图像副本以避免修改原始图像
        visualization_image = rgb_image.copy()
        
        # 确保保存目录存在
        ImageUtils.ensure_directory_exists(save_dir)
        
        # 遍历target3d中的每个目标
        for obj in target3d:
            # print(f"处理目标: {obj.get('label', 'Unknown')}, ID: {obj.get('id', 'N/A')}")
            
            # 绘制边界框
            if 'bbox' in obj and len(obj['bbox']) >= 2:
                # 边界框坐标
                pt1 = tuple(map(int, obj['bbox'][0]))
                pt2 = tuple(map(int, obj['bbox'][1]))
                print(f"绘制边界框: {pt1} 到 {pt2}")
                # 绘制边界框
                cv2.rectangle(visualization_image, pt1, pt2, (0, 255, 0), 2)
                
                # 显示标签和ID
                label_text = f"ID:{obj.get('id', 'N/A')} {obj.get('label', 'Unknown')}"
                cv2.putText(visualization_image, label_text, (pt1[0], pt1[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制质心点
            if 'center' in obj and len(obj['center']) == 2:
                center = tuple(map(int, obj['center']))
                print(f"绘制质心: {center}")
                cv2.circle(visualization_image, center, 5, (255, 0, 0), -1)
                # 显示质心坐标
                cv2.putText(visualization_image, f"{center}", (center[0]+10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 绘制随机点
            if 'npoints' in obj:
                print(f"绘制随机点数量: {len(obj['npoints'])}")
                for point in obj['npoints']:
                    if len(point) == 2:
                        point_coord = tuple(map(int, point))
                        cv2.circle(visualization_image, point_coord, 2, (0, 0, 255), -1)
        
        # 构建保存路径
        visualization_path = os.path.join(save_dir, filename)
        
        # 保存可视化结果
        success = ImageUtils.save_image(visualization_image, visualization_path)
        
        if success:
            print(f"Target3D可视化结果已保存到: {visualization_path}")
        
        return visualization_path