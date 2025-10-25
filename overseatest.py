from google import genai
from google.genai import types
import time

import cv2
import time
import threading
import queue
import random
from google import genai
from google.genai import types
import numpy as np
import json
import re

# 配置模型参数
# MODEL_ID = "gemini-robotics-er-1.5-preview" 
MODEL_ID = "gemini-2.5-flash"
ABIN_API_KEY = "AIzaSyCAWXAVLmpUb5dMPfuPjCwsoXMAYiHPqKc"
ZZZ_API_KEY = "sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6"

# client = genai.Client(api_key=ABIN_API_KEY)
client = genai.Client(
    api_key=ZZZ_API_KEY,
    http_options={
        "base_url": "https://api.zhizengzeng.com/google"
    },
    )

# 全局变量用于异步处理
result_queue = queue.Queue()
is_processing = False
processing_thread = None

object = ["people"]
temperature = 0.5

PROMPT0 = f"""
          Point to {object} in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{{"point": <point>,
          "label": <label1>}}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """

PROMPT1 = f"""
      Return bounding boxes as a JSON array with labels. Never return masks
      or code fencing. Limit to these objects: {object}. Include as many objects as you
      can identify on the table.
      If an object is present multiple times, name them according to their
      unique characteristic (colors, size, position, unique characteristics, etc..).
      The format should be as follows: [{{"box_2d": [ymin, xmin, ymax, xmax],
      "label": <label for the object>}}] normalized to 0-1000. The values in
      box_2d must only be integers
      """

PROMPT = PROMPT1

def parse_json_response(response_text):
    """
    解析JSON格式的响应文本，处理可能包含在Markdown中的情况
    增强了对多物体检测结果的支持
    """
    print(f"原始响应文本: {response_text[:200]}...")  # 打印响应前200个字符用于调试
    
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # 尝试从Markdown代码块中提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                print("错误：无法解析Markdown代码块中的JSON")
        
        # 尝试去掉文本前后可能的Markdown标记或其他无关文本
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("*") or cleaned_text.startswith("#") or cleaned_text.startswith("\n"):
            # 处理可能的列表项或标题格式
            cleaned_text = re.sub(r'^[*#\s]+', '', cleaned_text)
        
        # 尝试再次直接解析清理后的文本
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"清理后文本: {cleaned_text[:200]}...")  # 打印清理后的文本用于调试
            
        # 尝试寻找任何JSON格式的内容
        try:
            # 查找第一个[开头，最后一个]结尾的内容
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                print(f"提取的JSON字符串: {json_str}")  # 打印提取的字符串用于调试
                return json.loads(json_str)
        except Exception as e:
            print(f"错误：无法解析提取的JSON - {str(e)}")
    
    print("警告：无法解析JSON响应，返回空列表")
    return []

def draw_detections_on_frame(frame, detections):
    """
    在图像上绘制检测结果，使用固定颜色
    detections: 可以是包含{"point": [y, x], "label": "..."}的列表（点格式）
    或者包含{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}的列表（边界框格式）
    """
    # 创建图像副本以避免修改原始图像
    result_frame = frame.copy()
    height, width = result_frame.shape[:2]
    
    # 使用固定颜色（蓝色）
    fixed_color = (255, 0, 0)  # BGR格式的蓝色
    
    # 处理每个检测结果
    for detection in detections:
        try:
            # 使用固定颜色
            color = fixed_color
            
            # 为文本背景生成稍微深一点的同色系颜色，保持视觉协调
            bg_r = max(0, color[0] - 30)
            bg_g = max(0, color[1] - 30)
            bg_b = max(0, color[2] - 30)
            bg_color = (bg_r, bg_g, bg_b)
            
            # 检查文本颜色的可见性，如果背景颜色太亮，使用黑色文本
            text_color = (0, 0, 0) if (bg_r + bg_g + bg_b > 380) else (255, 255, 255)
            
            # 检查是否是边界框格式
            if "box_2d" in detection and "label" in detection:
                # 边界框格式 [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = detection["box_2d"]
                
                # 转换为像素坐标
                x1 = int((xmin / 1000.0) * width)
                y1 = int((ymin / 1000.0) * height)
                x2 = int((xmax / 1000.0) * width)
                y2 = int((ymax / 1000.0) * height)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # 绘制边界框 - 使用随机颜色
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签文本
                label = detection["label"]
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = x1
                text_y = y1 - 10 if y1 > 30 else y1 + text_size[1] + 20
                
                # 绘制文本背景 - 使用协调的背景色
                cv2.rectangle(
                    result_frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    bg_color,
                    -1
                )
                
                # 绘制文本 - 根据背景亮度选择合适的文本颜色
                cv2.putText(
                    result_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2
                )
            # 检查是否是点格式
            elif "point" in detection and "label" in detection:
                # 获取归一化坐标 [y, x] 范围0-1000
                y_normalized, x_normalized = detection["point"]
                
                # 转换为像素坐标
                x_pixel = int((x_normalized / 1000.0) * width)
                y_pixel = int((y_normalized / 1000.0) * height)
                
                # 绘制圆表示检测点 - 使用随机颜色
                cv2.circle(result_frame, (x_pixel, y_pixel), 10, color, -1)
                
                # 绘制标签文本
                label = detection["label"]
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = x_pixel - text_size[0] // 2
                text_y = y_pixel - 20
                
                # 确保文本不会超出图像边界
                if text_x < 0:
                    text_x = 10
                if text_y < 0:
                    text_y = 30
                
                # 绘制文本背景 - 使用协调的背景色
                cv2.rectangle(
                    result_frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    bg_color,
                    -1
                )
                
                # 绘制文本 - 根据背景亮度选择合适的文本颜色
                cv2.putText(
                    result_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2
                )
        except Exception as e:
            print(f"绘制检测结果时出错: {str(e)}")
    
    return result_frame

def process_image_async(img_bytes):
    """异步处理图像的函数，在单独线程中执行"""
    global is_processing
    try:
        start_time = time.time()
        
        # 使用Client类进行API调用，确保兼容性
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/jpeg',
                ),
                PROMPT  
            ],
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=500,  # 增加输出长度以适应多个物体的边界框数据
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
        print(f"异步推理时间: {time.time() - start_time:.4f}秒")
        
        # 解析结果并放入队列
        if response and hasattr(response, 'text'):
            detections = parse_json_response(response.text)
            result_queue.put(detections)
    except Exception as e:
        print(f"异步API调用错误: {str(e)}")
    finally:
        is_processing = False

def main():
    # 打开USB摄像头（0通常是默认摄像头，多个摄像头可以尝试1, 2, 3等）
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头较低的分辨率以提高获取速度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已成功打开！按 'q' 键退出")
    
    # 初始化帧率计算变量
    fps = 0
    prev_time = 0
    
    # 更激进的优化参数
    api_call_interval = 15  # 大幅降低API调用频率，15帧才调用一次
    frame_count = 0
    cached_detections = None  # 缓存检测结果
    last_detection_time = 0
    detection_timeout = 3.0  # 检测结果3秒后过期
    
    # 更激进的图像压缩参数
    target_width = 320  # 进一步降低分辨率到320px
    jpeg_quality = 60  # 降低JPEG质量以获得更小的文件大小
    
    global is_processing, processing_thread
    
    try:
        while True:
            # 记录当前时间
            current_time = time.time()
            
            # 读取一帧图像
            ret, frame = cap.read()
            frame_count += 1
            
            # 检查是否有新的检测结果可用
            if not result_queue.empty():
                cached_detections = result_queue.get()
                last_detection_time = current_time
            
            # 检查检测结果是否过期
            if current_time - last_detection_time > detection_timeout:
                cached_detections = None
            
            # 只有在指定间隔且没有正在处理的任务时才调用API
            if frame_count % api_call_interval == 0 and not is_processing:
                is_processing = True
                
                # 降低图像分辨率
                height, width = frame.shape[:2]
                aspect_ratio = height / width
                target_height = int(target_width * aspect_ratio)
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # 压缩图像并转换为bytes，使用更低的质量
                _, img_encoded = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                img_bytes = img_encoded.tobytes()
                
                # 启动异步处理线程
                if processing_thread and processing_thread.is_alive():
                    processing_thread.join(timeout=0.1)
                
                processing_thread = threading.Thread(target=process_image_async, args=(img_bytes,))
                processing_thread.daemon = True
                processing_thread.start()
            
            # 使用缓存的检测结果绘制（即使不是当前帧的结果）
            if cached_detections:
                frame = draw_detections_on_frame(frame, cached_detections)
            # 检查是否成功读取图像
            if not ret:
                print("错误：无法获取图像帧")
                break
            
            # 计算帧率
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            
            # 在画面上显示帧率
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示图像
            cv2.imshow('USB Camera', frame)
            
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户退出")
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 释放摄像头和关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已释放，程序退出")

if __name__ == "__main__":
    main()