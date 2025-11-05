from google import genai
from google.genai import types
import cv2
import numpy as np
import json
import time


class LLM:
    """
    语言模型类，用于文本生成和理解
    """
    
    def __init__(self, model_id="gemini-2.5-flash", api_key="sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6", 
                 base_url="https://api.zhizengzeng.com/google"):
        """
        初始化语言模型
        
        Args:
            model_id: 使用的模型ID
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_id = model_id
        self.client = genai.Client(
            api_key=api_key,
            http_options={"base_url": base_url}
        )
    
    def generate_text(self, prompt, temperature=0.5, verbose=False):
        """
        生成文本响应
        
        Args:
            prompt: 输入提示文本
            temperature: 生成内容的温度参数
            verbose: 是否打印详细信息
            
        Returns:
            生成的文本
        """
        if verbose:
            print("开始文本生成")
            start_time = time.time()
        
        # 调用模型生成文本
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        if verbose:
            print(f"生成时间: {time.time() - start_time:.2f} 秒")
        
        return response.text


class VLM:
    """
    视觉语言模型类，用于图像处理和理解
    """
    
    def __init__(self, model_id="gemini-2.5-flash", api_key="sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6", 
                 base_url="https://api.zhizengzeng.com/google"):
        """
        初始化视觉语言模型
        
        Args:
            model_id: 使用的模型ID
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_id = model_id
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
        _, img_encoded = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return img_encoded.tobytes()
    
    def analyze_image(self, rgb_image, prompt, temperature=0.5, jpeg_quality=60, verbose=False):
        """
        分析图像并根据提示生成响应
        
        Args:
            rgb_image: RGB格式的图像数据
            prompt: 分析提示文本
            temperature: 生成内容的温度参数
            jpeg_quality: JPEG编码质量
            verbose: 是否打印详细信息
            
        Returns:
            模型生成的响应文本
        """
        if verbose:
            print("开始图像分析")
            start_time = time.time()
        
        # 准备图像数据
        image_bytes = self.prepare_image(rgb_image, jpeg_quality)
        
        # 调用模型进行分析
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        if verbose:
            print(f"分析时间: {time.time() - start_time:.2f} 秒")
        
        return response.text
    
    def parse_json_response(self, response_text, verbose=False):
        """
        解析JSON格式的响应文本
        
        Args:
            response_text: 模型返回的文本
            verbose: 是否打印解析结果
            
        Returns:
            解析后的JSON数据
        """
        # 清理文本，移除可能的代码块标记
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # 解析JSON
        result = json.loads(cleaned_text)
        
        if verbose:
            print(result)
        
        return result
    
    def denormalize_coordinates(self, detection_result, image_shape, verbose=False):
        """
        将归一化的坐标（0-1000范围）转换为实际像素坐标
        
        Args:
            detection_result: 包含归一化坐标的检测结果（JSON格式或字典列表）
            image_shape: 图像形状 (height, width)
            verbose: 是否打印转换结果
            
        Returns:
            转换后的检测结果，包含实际像素坐标
        """
        # 确保detection_result是列表格式
        if not isinstance(detection_result, list):
            # 如果传入的是JSON字符串，先解析
            if isinstance(detection_result, str):
                # 清理文本，移除可能的代码块标记
                cleaned_text = detection_result.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                detection_result = json.loads(cleaned_text)
            else:
                raise TypeError("detection_result必须是列表或JSON字符串格式")
        
        # 获取图像高度和宽度
        height, width = image_shape
        
        # 复制结果以避免修改原始数据
        result = []
        for detection in detection_result:
            detection_copy = detection.copy()
            
            # 处理边界框坐标
            if "box_2d" in detection_copy:
                box_2d = detection_copy["box_2d"]
                # 验证box_2d格式
                if isinstance(box_2d, list) and len(box_2d) >= 4:
                    # 取前4个值作为[ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = box_2d[:4]
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
                    detection_copy["box_2d"] = [y1, x1, y2, x2]
                else:
                    if verbose:
                        print(f"警告: box_2d格式不正确，期望包含至少4个值，但得到: {box_2d}")
                    # 如果格式不正确，跳过此检测结果
                    continue
            
            # 处理点坐标（如果存在）
            elif "point" in detection_copy:
                # 点坐标是[y, x]格式，归一化到0-1000
                y_norm, x_norm = detection_copy["point"]
                # 转换为真实像素坐标
                x = int((x_norm / 1000.0) * width)
                y = int((y_norm / 1000.0) * height)
                # 确保坐标在有效范围内
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                detection_copy["point"] = [y, x]
            
            result.append(detection_copy)
        
        if verbose:
            print("反归一化后的坐标结果:", result)
        
        return result


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
    
    # 1. 使用LLM类进行文本生成
    llm = LLM()
    text_response = llm.generate_text(
        prompt="解释什么是视觉语言模型",
        verbose=True
    )
    print("\nLLM响应:", text_response)
    
    # 2. 使用VLM类进行图像分析
    vlm = VLM()
    
    # 简单的图像描述
    description_prompt = "详细描述这张图片中包含的内容"
    description_response = vlm.analyze_image(
        rgb_image=rgb_image,
        prompt=description_prompt,
        verbose=True
    )
    print("\nVLM图像描述:", description_response)
    
    # 对象检测（点坐标）
    point_detection_prompt = """
    Point to the main object in the image. The label returned
    should be an identifying name for the object detected.
    The answer should follow the json format: [{"point": <point>,
    "label": <label>}]. The points are in [y, x] format
    normalized to 0-1000.
    """
    point_detection_response = vlm.analyze_image(
        rgb_image=rgb_image,
        prompt=point_detection_prompt,
        verbose=True
    )
    
    # 解析JSON响应
    point_data = vlm.parse_json_response(point_detection_response)
    print("\nVLM点检测结果:", point_data)
    
    # 3. 传统的ImageObjectDetector类仍可使用
    detector = ImageObjectDetector(detection_type="point")
    detections = detector.detect_objects(
        rgb_image=rgb_image,
        control_instruction="chair",
        verbose=True
    )
    print("\n对象检测器结果:", detections)