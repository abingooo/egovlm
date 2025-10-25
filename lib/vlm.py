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
            detection_type: 检测类型，"point"或"box"，默认为"point"
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
        创建检测提示词
        
        Args:
            objects_to_detect: 要检测的对象列表
            prompt_type: 提示词类型，"point"或"box"
            
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
        else:  # prompt_type == "box"
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
    
    def detect_objects(self, rgb_image, objects_to_detect, 
                      temperature=0.5, jpeg_quality=60, verbose=True):
        """
        检测图像中的对象
        
        Args:
            rgb_image: RGB格式的图像数据
            objects_to_detect: 要检测的对象列表
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
        prompt = self.create_prompt(objects_to_detect, self.detection_type)
        
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
                    # 点坐标是[y, x]格式，归一化到0-1000
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

# # 使用示例
# if __name__ == "__main__":
#     # 导入日志读取模块
#     from lib.logread import create_reader
    
#     # 创建日志读取器
#     reader = create_reader('./log')
    
#     # 读取最新的日志数据
#     latest_data = reader.read_latest_log()
    
#     # 获取RGB图像
#     rgb_image = latest_data['color_image']
    
#     # 创建对象检测器
#     detector = ImageObjectDetector()
    
#     # 检测对象（设置verbose=False避免重复打印）
#     detections = detector.detect_objects(
#         rgb_image=rgb_image,
#         objects_to_detect=["Trash can","the chair on the right"],
#         prompt_type="point",
#         verbose=False
#     )
#     print(detections)