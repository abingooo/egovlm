from google import genai
from google.genai import types
from openai import OpenAI
import cv2
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
        if 'gemini' in model_id:
            self.client = genai.Client(
                api_key=api_key,
                http_options={"base_url": base_url}
            )
        else:
            self.client = OpenAI(
            # openai系列的sdk，包括langchain，都需要这个/v1的后缀
            base_url='https://api.zhizengzeng.com/v1',
            api_key='sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6',
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
        if 'gemini' in self.model_id:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            response = response.text

        else:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_id,
            )
            response = response.choices[0].message.content
        
        if verbose:
            print(f"生成时间: {time.time() - start_time:.2f} 秒")
        
        return response

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
