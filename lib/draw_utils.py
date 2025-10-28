"""
draw_utils.py - 图像绘制工具类

提供用于绘制检测结果的工具函数，包括点标记和边界框绘制等功能。
"""

import cv2
import numpy as np


class DrawUtils:
    """
    图像绘制工具类，用于在图像上绘制检测结果
    """
    
    def __init__(self):
        """
        初始化绘制工具类
        """
        # 默认颜色配置
        self.colors = {
            'point': (0, 255, 0),  # 绿色点
            'box': (255, 0, 0),    # 蓝色边界框
            'text': (255, 255, 255),  # 白色文本
            'bg': (0, 0, 0)        # 黑色文本背景
        }
    
    def draw_point_detection(self, image, detections):
        """
        在图像上绘制点检测结果
        
        Args:
            image: 输入图像（BGR格式）
            detections: 点检测结果列表，格式为[{"point": [y, x], "label": "label_name"}, ...]，
                       其中point已经是真实像素坐标
            
        Returns:
            绘制后的图像
        """
        # 创建图像副本以避免修改原始图像
        result_image = image.copy()
        height, width = result_image.shape[:2]
        
        for detection in detections:
            try:
                # 获取点坐标和标签
                if "point" in detection:
                    # 点坐标已经是真实像素坐标[y, x]
                    y, x = detection["point"]
                    
                    # 确保坐标在有效范围内
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    
                    # 绘制点（使用圆形表示）
                    cv2.circle(result_image, (x, y), 10, self.colors['point'], -1)  # 填充圆
                    cv2.circle(result_image, (x, y), 12, (0, 0, 0), 2)  # 黑色边框
                    
                    # 绘制标签
                    if "label" in detection:
                        label = detection["label"]
                        self._draw_label(result_image, label, x, y)
            except Exception as e:
                print(f"绘制点检测结果时出错: {str(e)}")
        
        return result_image
    
    def draw_box_detection(self, image, detections):
        """
        在图像上绘制边界框检测结果
        
        Args:
            image: 输入图像（BGR格式）
            detections: 边界框检测结果列表，格式为[{"box_2d": [ymin, xmin, ymax, xmax], "label": "label_name"}, ...]，
                       其中box_2d已经是真实像素坐标
            
        Returns:
            绘制后的图像
        """
        # 创建图像副本以避免修改原始图像
        result_image = image.copy()
        height, width = result_image.shape[:2]
        
        for detection in detections:
            try:
                # 获取边界框坐标和标签
                if "box_2d" in detection:
                    # 边界框坐标已经是真实像素坐标[ymin, xmin, ymax, xmax]
                    y1, x1, y2, x2 = detection["box_2d"]
                    
                    # 确保坐标在有效范围内
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # 绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), self.colors['box'], 2)
                    
                    # 绘制标签
                    if "label" in detection:
                        label = detection["label"]
                        # 将标签绘制在边界框上方
                        self._draw_label(result_image, label, x1, y1 - 10)
            except Exception as e:
                print(f"绘制边界框检测结果时出错: {str(e)}")
        
        return result_image
    
    def _draw_label(self, image, label, x, y):
        """
        在图像上绘制文本标签
        
        Args:
            image: 输入图像
            label: 标签文本
            x: 标签起始x坐标
            y: 标签起始y坐标
        """
        height = image.shape[0]
        
        # 获取文本大小
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 调整文本位置，确保不超出图像边界
        text_y = y
        if y < 30:
            text_y = y + text_size[1] + 20  # 如果位置太靠上，放在下方
        elif y > height - 30:
            text_y = height - 30  # 如果位置太靠下，放在安全区域
        
        # 绘制文本背景
        cv2.rectangle(
            image,
            (x - 5, text_y - text_size[1] - 5),
            (x + text_size[0] + 5, text_y + 5),
            self.colors['bg'],
            -1
        )
        
        # 绘制文本
        cv2.putText(
            image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors['text'],
            2
        )
    
    def draw_detections(self, image, detections, detection_type="point"):
        """
        根据检测类型绘制结果
        
        Args:
            image: 输入图像（BGR格式）
            detections: 检测结果列表
            detection_type: 检测类型，"point"或"box"
            
        Returns:
            绘制后的图像
        """
        if "point" in detection_type :
            return self.draw_point_detection(image, detections)
        elif "box" in detection_type :
            return self.draw_box_detection(image, detections)
        else:
            print(f"不支持的检测类型: {detection_type}")
            return image.copy()
    
    def show_image(self, image, window_name="Detection Result"):
        """
        显示图像
        
        Args:
            image: 要显示的图像
            window_name: 窗口名称（建议使用英文以避免乱码）
        """
        # 使用英文标题避免Windows系统下的中文乱码问题
        display_name = window_name
        
        cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
        cv2.imshow(display_name, image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_image(self, image, file_path):
        """
        保存图像到文件
        
        Args:
            image: 要保存的图像
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            cv2.imwrite(file_path, image)
            print(f"图像已保存至: {file_path}")
            return True
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return False


# 便捷函数
def draw_and_display(image, detections, detection_type="point", save_path=None):
    """
    绘制检测结果并显示/保存图像的便捷函数
    
    Args:
        image: 输入图像（BGR格式）
        detections: 检测结果列表
        detection_type: 检测类型，"point"或"box"
        save_path: 保存路径，如果为None则不保存
        
    Returns:
        绘制后的图像
    """
    # 创建绘制工具实例
    drawer = DrawUtils()
    
    # 绘制检测结果
    result_image = drawer.draw_detections(image, detections, detection_type)
    
    # 保存图像（如果指定了路径）
    if save_path:
        drawer.save_image(result_image, save_path)
    
    # 显示图像
    drawer.show_image(result_image)
    
    return result_image


# 便捷函数
def draw_and_save(image, detections, detection_type="point", save_path=None):
    """
    绘制检测结果并保存图像的便捷函数
    
    Args:
        image: 输入图像（BGR格式）
        detections: 检测结果列表
        detection_type: 检测类型，"point"或"box"
        save_path: 保存路径，如果为None则不保存
        
    Returns:
        绘制后的图像
    """
    # 创建绘制工具实例
    drawer = DrawUtils()
    
    # 绘制检测结果
    result_image = drawer.draw_detections(image, detections, detection_type)
    
    # 保存图像（如果指定了路径）
    if save_path:
        drawer.save_image(result_image, save_path)
    
    return result_image