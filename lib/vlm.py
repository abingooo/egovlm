from google import genai
from google.genai import types
import cv2
import numpy as np
import json
import time


class ImageObjectDetector:
    """
    图像对象检测器类，用于检测图像中的指定对象
    """
    
    def __init__(self, model_id="gemini-2.5-flash", api_key="sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6", 
                 base_url="https://api.zhizengzeng.com/google", detection_type="point"):
        """
        初始化图像对象检测器
        
        Args:
            model_id: 使用的模型ID
            api_key: API密钥
            base_url: API基础URL
            detection_type: 检测类型，"point"、"box"、"flypoint"或"flybox"，默认为"point"
        """
        self.model_id = model_id
        self.detection_type = detection_type
        self.client = genai.Client(
            api_key=api_key,
            http_options={"base_url": base_url}
        )
    
    def prepare_image(self, rgb_image, jpeg_quality=60):
        """
        准备图像数据，将RGB图像编码为JPEG格式
        
        Args:
            rgb_image: RGB格式的图像数据
            jpeg_quality: JPEG编码质量
            
        Returns:
            编码后的图像字节数据
        """
        # 将图像编码为JPEG格式并获取字节数据
        _, img_encoded = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return img_encoded.tobytes()
    
    def create_prompt(self, objects_to_detect, prompt_type="point"):
        """
        创建检测提示词prompt
        
        Args:
            objects_to_detect: 要检测的对象列表或控制指令
            prompt_type: 提示词类型，"point"、"box"、"flypoint"或"flybox"
            
        Returns:
            格式化的提示词
        """
        if prompt_type == "point":
            return f"""
              Point to {objects_to_detect} in the image. The label returned
              should be an identifying name for the object detected.
              The answer should follow the json format: [{{"point": <point>,
              "label": <label1>}}, ...]. The points are in [y, x] format
              normalized to 0-1000.
            """
        elif prompt_type == "box":
            return f"""
          Return bounding boxes as a JSON array with labels. Never return masks
          or code fencing. Limit to these objects: {objects_to_detect}. Include as many objects as you
          can identify.
          If an object is present multiple times, name them according to their
          unique characteristic (colors, size, position, unique characteristics, etc..).
          The format should be as follows: [{{"box_2d": [ymin, xmin, ymax, xmax],
          "label": <label for the object>}}] normalized to 0-1000. The values in
          box_2d must only be integers
          """
        elif prompt_type == "flypoint":
            return f"""
              Analyze the drone control instruction: '{objects_to_detect}'
              
              1. Extract the target object mentioned in the instruction
              2. Identify the specific object in the image if there are multiple similar objects
                 (based on description like "on the left", "larger", etc.)
              3. Point to the center of the target object in the image
              
              Return the answer as a JSON array with the following format:
              [{{"point": [y, x], "label": "<target_object>"}}] where:
              - point is in [y, x] format normalized to 0-1000
              - label is the extracted target object name
              
              Never return any additional text, markdown formatting, or explanations.
              Only return the JSON array as specified.
            """
        elif prompt_type == "flybox":
            return f"""
              Analyze the drone control instruction: '{objects_to_detect}'
              
              1. Extract the target object mentioned in the instruction
              2. Identify the specific object in the image if there are multiple similar objects
                 (based on description like "on the left", "larger", "blue", etc.)
              3. Draw a bounding box around the entire target object
              
              Return the answer as a JSON array with the following format:
              [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "<target_object>"}}] where:
              - box_2d is [ymin, xmin, ymax, xmax] format normalized to 0-1000
              - All values in box_2d must be integers
              - label is the extracted target object name
              
              Never return any additional text, markdown formatting, or explanations.
              Only return the JSON array as specified.
            """
        
        elif prompt_type == "waypoint":
            return f"""
        Act as a drone flight path planner analyzing images from the drone's front-facing camera.
        
        The image you see is captured by the drone's camera.
        
        The target object has the following 3D coordinates in the camera coordinate system (in meters):
        - x: forward direction (positive values mean the object is ahead of the camera)
        - y: left direction (positive values mean the object is to the left of the camera)
        - z: upward direction (positive values mean the object is above the camera)
        
        IMPORTANT: All physical objects have volume. When planning the flight path, you must consider the object's size and maintain a safe distance (at least 0.5 meters) from the object at all times.
        
        For common objects:
        - Water dispenser: approximately 0.3-0.4m in width, 0.3-0.4m in depth, 1.2-1.8m in height
        - Furniture: typically 0.4-1.0m in width/depth, 0.4-2.0m in height
        - Other objects: estimate their dimensions based on visual appearance
        
        Your task is to plan a safe flight path from the drone's current position to complete the control instruction: '{objects_to_detect}'.
        
        Please generate waypoints in 3D space coordinates (in meters) that will guide the drone from its starting position to complete the control instruction.
        The waypoints should be generated based on the visual scene and the target object's marked position.
        
        Please ensure:
        - The trajectory formed by all points is smooth and allows the drone to fly safely
        - Maintain a minimum safety distance of 0.5 meters from all objects, including the target object
        - Avoid obstacles present in the scene when planning the path
        - The waypoints are reasonably distributed, neither too dense nor too sparse
        - The path follows the most direct route while ensuring safety
        - The path should be practical for real drone flight based on the camera's perspective
        - The first waypoint should be the current position of the drone
        - The last waypoint should be at a safe distance from the target object, never inside its estimated volume
        The answer must follow this JSON format:
        [{{"point": [y, x, z], "label": "<index>"}}, ...]
        where:
        - point is a 3D coordinate [y, x, z] in meters, following the camera coordinate system
        - label is the corresponding point index, numbered sequentially starting from '0'
        
        Please return only the JSON formatted answer without any additional text or explanation.
          """
        elif prompt_type == "waypoint1":
            return f"""
        Act as a drone flight path planner analyzing images from the drone's front-facing camera.
        
        The image you see is captured by the drone's camera.
        
        The target object has the following 3D coordinates in the camera coordinate system (in meters):
        - x: forward direction (positive values mean the object is ahead of the camera)
        - y: left direction (positive values mean the object is to the left of the camera)
        - z: upward direction (positive values mean the object is above the camera)
        
        IMPORTANT: All physical objects have volume. When planning the flight path, you must consider the object's size and maintain a safe distance from the object at all times.
        
        For common objects:
        - Water dispenser: approximately 0.3-0.4m in width, 0.3-0.4m in depth, 1.2-1.8m in height
        - Furniture: typically 0.4-1.0m in width/depth, 0.4-2.0m in height
        - Other objects: estimate their dimensions based on visual appearance
        
        Your task is to plan a safe flight path from the drone's current position to complete the control instruction: '{objects_to_detect}'.
        
        Please generate waypoints in 3D space coordinates (in meters) that will guide the drone from its starting position to complete the control instruction.
        The waypoints should be generated based on the visual scene and the target object's marked position.
        
        IMPORTANT: Analyze the instruction carefully:
        - If the instruction contains "around" or "circle", this means the drone should fly in a circular path around the target object
        - For such circling instructions, create waypoints that form a loop around the target object, maintaining a minimum safety distance of 1.0 meter from the object's estimated boundaries
        - Ensure the flight path smoothly transitions from approaching the target to circling around it
        - After circling, you may optionally include a return path back to a starting position if appropriate
        
        Please ensure:
        - The trajectory formed by all points is smooth and allows the drone to fly safely
        - Maintain a minimum safety distance of 0.5 meters from all objects for "fly to" instructions, and 1.0 meter for "fly around" instructions
        - When estimating object boundaries, add extra safety margin to avoid collisions
        - Avoid obstacles present in the scene when planning the path
        - The waypoints are reasonably distributed, neither too dense nor too sparse
        - The path follows the most direct route while ensuring safety
        - The path should be practical for real drone flight based on the camera's perspective
        - For "fly to" instructions: The first waypoint should be the current position of the drone, and the last waypoint should be at a safe distance from the target object
        - For "fly around" instructions: Include waypoints for approaching the target, circling the target (at least 4 points to form a circular path while maintaining safe distance), and any necessary exit path
        The answer must follow this JSON format:
        [{{"point": [y, x, z], "label": "<index>"}}, ...]
        where:
        - point is a 3D coordinate [y, x, z] in meters, following the camera coordinate system
        - label is the corresponding point index, numbered sequentially starting from '0'
        
        Please return only the JSON formatted answer without any additional text or explanation.
          """
    
    def detect_objects(self, rgb_image, control_instruction, 
                      temperature=0.5, jpeg_quality=60, verbose=True):
        """
        检测图像中的对象，支持从控制指令中提取目标
        
        Args:
            rgb_image: RGB格式的图像数据
            control_instruction: 控制指令字符串（如"fly to the chair on the left"）
                                或传统的对象名称列表
            temperature: 生成内容的温度参数
            jpeg_quality: JPEG编码质量
            verbose: 是否打印详细信息
            
        Returns:
            检测结果（字典或列表）
        """
        if verbose:
            print("开始推理")
            start_time = time.time()
        
        # 准备图像数据
        image_bytes = self.prepare_image(rgb_image, jpeg_quality)
        
        # 创建提示词，使用实例变量detection_type
        prompt = self.create_prompt(control_instruction, self.detection_type)
        
        # 调用模型进行推理
        image_response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ],
            config = types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        if verbose:
            print(f"推理时间: {time.time() - start_time:.2f} 秒")
        
        # 解析结果，传入图像形状以转换为真实像素坐标
        image_shape = rgb_image.shape[:2]  # 获取图像高度和宽度
        return self.parse_response(image_response.text, image_shape=image_shape, verbose=verbose)
    
    def parse_response(self, response_text, image_shape=None, verbose=False):
        """
        解析模型返回的文本响应
        
        Args:
            response_text: 模型返回的文本
            image_shape: 图像形状 (height, width)，如果提供则将坐标转换为真实像素坐标
            verbose: 是否打印解析结果
            
        Returns:
            解析后的JSON数据，包含真实像素坐标
        """
        # 清理文本，移除可能的代码块标记
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # 解析JSON
        detections = json.loads(cleaned_text)
        
        # 如果提供了图像形状，将归一化坐标转换为真实像素坐标
        if image_shape is not None:
            height, width = image_shape
            for detection in detections:
                if "point" in detection:
                    # 根据detection_type判断是2D坐标还是3D坐标
                    if hasattr(self, 'detection_type') and (self.detection_type == "waypoint" or self.detection_type == "waypoint1"):
                        # 3D坐标格式 [y, x, z]，单位为米，不需要转换
                        # 直接保留原始3D坐标
                        pass  # 3D坐标不需要转换为像素坐标
                    else:
                        # 2D坐标是[y, x]格式，归一化到0-1000
                        y_norm, x_norm = detection["point"]
                        # 转换为真实像素坐标
                        x = int((x_norm / 1000.0) * width)
                        y = int((y_norm / 1000.0) * height)
                        # 确保坐标在有效范围内
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        detection["point"] = [y, x]
                elif "box_2d" in detection:
                    # 边界框坐标是[ymin, xmin, ymax, xmax]格式，归一化到0-1000
                    ymin, xmin, ymax, xmax = detection["box_2d"]
                    # 转换为真实像素坐标
                    x1 = int((xmin / 1000.0) * width)
                    y1 = int((ymin / 1000.0) * height)
                    x2 = int((xmax / 1000.0) * width)
                    y2 = int((ymax / 1000.0) * height)
                    # 确保坐标在有效范围内
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    detection["box_2d"] = [y1, x1, y2, x2]
        
        if verbose:
            print(detections)
        
        return detections

# 使用示例
if __name__ == "__main__":
    # 导入日志读取模块
    from lib.logread import create_reader
    
    # 创建日志读取器
    reader = create_reader('./log')
    
    # 读取最新的日志数据
    latest_data = reader.read_latest_log()
    
    # 获取RGB图像
    rgb_image = latest_data['color_image']
    
    # 创建无人机控制模式的对象检测器（使用flypoint类型）
    drone_detector = ImageObjectDetector(detection_type="flypoint")
    
    # 使用控制指令进行检测
    detections = drone_detector.detect_objects(
        rgb_image=rgb_image,
        control_instruction="fly to the chair on the left",
        verbose=True
    )
    print("无人机控制指令检测结果:", detections)
    
    # 也可以使用边界框检测
    box_detector = ImageObjectDetector(detection_type="flybox")
    box_detections = box_detector.detect_objects(
        rgb_image=rgb_image,
        control_instruction="Fly around the tree in circles",
        verbose=True
    )
    print("无人机控制指令边界框检测结果:", box_detections)
    
    # 仍然支持传统的对象检测方式
    traditional_detector = ImageObjectDetector(detection_type="point")
    traditional_detections = traditional_detector.detect_objects(
        rgb_image=rgb_image,
        control_instruction="chair",  # 这里传入的是对象名称
        verbose=False
    )
    print("传统对象检测结果:", traditional_detections)