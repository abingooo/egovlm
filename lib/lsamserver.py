from PIL import Image
import numpy as np
import json
import os
import time
import torch
import sys
import cv2
import random
import logging
import argparse
from flask import Flask, request, jsonify
import io
from base64 import b64encode, b64decode

# 添加lsam目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lsam'))

# 导入LangSAM
from lang_sam import LangSAM

# 确保日志目录存在
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "lsamserver.log")),  # 记录到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger("lsamserver")

app = Flask(__name__)

# 设置环境变量和优化参数
def setup_environment(gpu_device=None):
    """设置环境变量和性能优化参数"""
    logger.info("开始设置环境...")
    os.environ["HF_HOME"] = "./model_cache"
    os.makedirs("./model_cache", exist_ok=True)
    logger.info(f"模型缓存目录设置为: {os.path.abspath('./model_cache')}")
    
    # 确保日志目录存在
    global log_dir
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"日志目录设置为: {os.path.abspath(log_dir)}")
    
    # 设置GPU设备
    if torch.cuda.is_available():
        device_id = 0
        # 优先使用传入的参数
        if gpu_device is not None:
            device_id = gpu_device
        else:
            # 其次使用环境变量
            env_device = os.environ.get('GPU_DEVICE')
            if env_device is not None:
                try:
                    device_id = int(env_device)
                except ValueError:
                    logger.warning(f"环境变量GPU_DEVICE={env_device}不是有效的整数，使用默认设备0")
        
        # 验证设备是否有效
        if device_id >= 0 and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            logger.info(f"使用GPU设备: {device_id} ({torch.cuda.get_device_name(device_id)})")
        else:
            logger.warning(f"GPU设备 {device_id} 不存在，共{torch.cuda.device_count()}个GPU可用，使用默认设备0")
            torch.cuda.set_device(0)
            logger.info(f"使用默认GPU设备: 0 ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("CUDA不可用，使用CPU进行计算")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, os.sched_getaffinity(0))
        logger.info(f"设置CPU亲和性: {len(os.sched_getaffinity(0))}个核心")
    logger.info("环境设置完成")

def create_timestamp_directory():
    """创建带时间戳的目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(log_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"创建时间戳目录: {timestamp_dir}")
    return timestamp_dir

# 全局模型实例
model_instance = None

def get_model():
    """获取LangSAM模型实例"""
    global model_instance
    if model_instance is None:
        logger.info("正在加载LangSAM模型...")
        start_time = time.time()
        try:
            model_instance = LangSAM()
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f} 秒")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    return model_instance

def process_mask_and_extract_features(masks):
    """处理掩码，提取质心、边界框和随机点特征"""
    detection_data = {
        "masks": []
    }
    
    for mask_idx, mask in enumerate(masks):
        binary_mask = (mask > 0).astype(np.uint8) * 255
        area = int(np.sum(mask > 0))
        
        if area > 0:
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 基于整个掩码计算边界框
            y_coords, x_coords = np.where(binary_mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_min, y_min = int(x_coords.min()), int(y_coords.min())
                x_max, y_max = int(x_coords.max()), int(y_coords.max())
                box_x1, box_y1 = x_min, y_min
                box_x2, box_y2 = x_max, y_max
                w = box_x2 - box_x1 + 1
                h = box_y2 - box_y1 + 1
                
                # 质心计算基于最大连通块
                centroid_x, centroid_y = 0, 0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    if cv2.contourArea(largest_contour) > 0:
                        M = cv2.moments(largest_contour)
                        if M['m00'] > 0:
                            centroid_x = int(M['m10'] / M['m00'])
                            centroid_y = int(M['m01'] / M['m00'])
                        else:
                            centroid_x = int((box_x1 + box_x2) / 2)
                            centroid_y = int((box_y1 + box_y2) / 2)
                    
                    # 腐蚀操作和随机点采样
                    kernel = np.ones((9, 9), np.uint8)
                    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
                    
                    mask_coords = np.column_stack(np.where(eroded_mask > 0))
                    random_points = []
                    
                    if len(mask_coords) > 0:
                        points_needed = 9
                        all_points = [(x, y) for y, x in mask_coords]
                        
                        if len(all_points) >= points_needed:
                            random_points = random.sample(all_points, points_needed)
                        else:
                            random_points = all_points[:]
                    
                    # 收集掩码数据
                    mask_data = {
                        "mask_id": mask_idx,
                        "area": area,
                        "centroid": [centroid_x, centroid_y],
                        "bounding_box": {
                            "x1": box_x1,
                            "y1": box_y1,
                            "x2": box_x2,
                            "y2": box_y2,
                            "width": w,
                            "height": h
                        },
                        "random_points": [[int(x), int(y)] for x, y in random_points[:9]]
                    }
                    
                    detection_data["masks"].append(mask_data)
    
    return detection_data

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求的API端点"""
    request_id = time.strftime("%Y%m%d%H%M%S_") + str(random.randint(1000, 9999))
    logger.info(f"[{request_id}] 接收到预测请求")
    
    # 创建时间戳目录用于保存结果
    timestamp_dir = create_timestamp_directory()
    
    try:
        # 获取请求数据
        data = request.json
        if not data or 'image' not in data or 'text' not in data:
            logger.warning(f"[{request_id}] 请求参数不完整: 缺少image或text")
            return jsonify({"error": "缺少必要参数:image或text"}), 400
        
        # 解码Base64图像数据
        text_prompt = data['text']
        logger.info(f"[{request_id}] 处理请求: 文本提示='{text_prompt}'")
        
        image_data = b64decode(data['image'])
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"[{request_id}] 图像解码成功，分辨率: {image_pil.size[0]}x{image_pil.size[1]}")
        
        # 保存原始图像
        original_image_path = os.path.join(timestamp_dir, f"original_{request_id}.png")
        image_pil.save(original_image_path)
        logger.info(f"[{request_id}] 原始图像已保存到: {original_image_path}")
        
        # 获取模型
        model = get_model()
        
        # 执行推理
        start_time = time.time()
        results = model.predict([image_pil], [text_prompt])
        inference_time = time.time() - start_time
        logger.info(f"[{request_id}] 推理完成，耗时: {inference_time:.2f} 秒")
        
        # 处理结果
        if results and len(results) > 0 and "masks" in results[0] and len(results[0]["masks"]) > 0:
            result = results[0]
            mask_count = len(result["masks"])
            logger.info(f"[{request_id}] 检测到 {mask_count} 个对象")
            
            detection_data = process_mask_and_extract_features(result["masks"])
            
            detection_data["image_path"] = "request_image"
            detection_data["text_prompt"] = text_prompt
            detection_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            detection_data["mask_count"] = len(detection_data["masks"])
            detection_data["request_id"] = request_id
            detection_data["save_dir"] = timestamp_dir
            
            # 添加可视化结果保存
            image_array = np.array(image_pil)
            
            # 定义颜色
            centroid_color = (0, 0, 255)  # 红色
            box_color = (0, 255, 0)       # 绿色
            random_color = (255, 165, 0)  # 橙色
            
            # 创建可视化图像
            visualized = image_array.copy()
            
            # 绘制检测结果
            for mask_data in detection_data["masks"]:
                # 边界框
                bbox = mask_data["bounding_box"]
                cv2.rectangle(visualized, 
                            (bbox["x1"], bbox["y1"]), 
                            (bbox["x2"], bbox["y2"]), 
                            box_color, 2)
                
                # 质心
                centroid = mask_data["centroid"]
                cv2.circle(visualized, (centroid[0], centroid[1]), 5, centroid_color, -1)
                cv2.putText(visualized, f'C{mask_data["mask_id"]}', (centroid[0] + 10, centroid[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, centroid_color, 1)
                
                # 随机点
                for idx, point in enumerate(mask_data["random_points"]):
                    cv2.circle(visualized, (point[0], point[1]), 4, random_color, -1)
                    cv2.putText(visualized, f'P{idx+1}', (point[0] + 8, point[1] + 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, random_color, 1)
            
            # 保存可视化结果
            output_image = Image.fromarray(np.uint8(visualized)).convert("RGB")
            visualized_path = os.path.join(timestamp_dir, f"visualized_{request_id}.png")
            output_image.save(visualized_path)
            logger.info(f"[{request_id}] 可视化结果已保存到: {visualized_path}")
            detection_data["visualized_path"] = visualized_path
            
            # 保存JSON结果
            json_path = os.path.join(timestamp_dir, f"result_{request_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[{request_id}] JSON结果已保存到: {json_path}")
            
            logger.info(f"[{request_id}] 请求处理成功")
            return jsonify(detection_data)
        else:
            logger.info(f"[{request_id}] 未检测到符合条件的对象")
            
            # 创建空结果
            empty_result = {
                "image_path": "request_image",
                "text_prompt": text_prompt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mask_count": 0,
                "masks": [],
                "request_id": request_id,
                "save_dir": timestamp_dir,
                "error": "未检测到符合条件的对象"
            }
            
            # 保存空结果JSON
            json_path = os.path.join(timestamp_dir, f"result_{request_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(empty_result, f, ensure_ascii=False, indent=2)
            logger.info(f"[{request_id}] 空结果JSON已保存到: {json_path}")
            
            return jsonify(empty_result)
            
    except Exception as e:
        logger.error(f"[{request_id}] 请求处理失败: {str(e)}")
        
        # 保存错误信息
        error_info = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }
        error_json_path = os.path.join(timestamp_dir, f"error_{request_id}.json")
        with open(error_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        logger.info(f"[{request_id}] 错误信息已保存到: {error_json_path}")
        
        return jsonify({"error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """处理可视化请求的API端点"""
    request_id = time.strftime("%Y%m%d%H%M%S_") + str(random.randint(1000, 9999))
    logger.info(f"[{request_id}] 接收到可视化请求")
    
    # 创建时间戳目录用于保存结果
    timestamp_dir = create_timestamp_directory()
    
    try:
        # 获取请求数据
        data = request.json
        if not data or 'image' not in data or 'result_json' not in data:
            logger.warning(f"[{request_id}] 请求参数不完整: 缺少image或result_json")
            return jsonify({"error": "缺少必要参数:image或result_json"}), 400
        
        # 解码Base64图像数据
        image_data = b64decode(data['image'])
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        result_json = data['result_json']
        mask_count = result_json.get("mask_count", 0)
        logger.info(f"[{request_id}] 图像解码成功，分辨率: {image_pil.size[0]}x{image_pil.size[1]}，需要可视化的掩码数: {mask_count}")
        
        # 保存原始图像
        original_image_path = os.path.join(timestamp_dir, f"visualize_original_{request_id}.png")
        image_pil.save(original_image_path)
        logger.info(f"[{request_id}] 可视化原始图像已保存到: {original_image_path}")
        
        # 进行可视化
        if mask_count > 0:
            start_time = time.time()
            image_array = np.array(image_pil)
            
            # 定义颜色
            centroid_color = (0, 0, 255)  # 红色
            box_color = (0, 255, 0)       # 绿色
            random_color = (255, 165, 0)  # 橙色
            
            # 创建可视化图像
            visualized = image_array.copy()
            
            # 绘制
            for mask_data in result_json["masks"]:
                # 边界框
                bbox = mask_data["bounding_box"]
                cv2.rectangle(visualized, 
                            (bbox["x1"], bbox["y1"]), 
                            (bbox["x2"], bbox["y2"]), 
                            box_color, 2)
                
                # 质心
                centroid = mask_data["centroid"]
                cv2.circle(visualized, (centroid[0], centroid[1]), 5, centroid_color, -1)
                cv2.putText(visualized, 'Centroid', (centroid[0] + 10, centroid[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, centroid_color, 1)
                
                # 随机点
                for idx, point in enumerate(mask_data["random_points"]):
                    cv2.circle(visualized, (point[0], point[1]), 4, random_color, -1)
                    cv2.putText(visualized, f'P{idx+1}', (point[0] + 8, point[1] + 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, random_color, 1)
            
            # 转换为PIL图像
            output_image = Image.fromarray(np.uint8(visualized)).convert("RGB")
            
            # 保存可视化结果
            visualized_path = os.path.join(timestamp_dir, f"visualized_{request_id}.png")
            output_image.save(visualized_path)
            logger.info(f"[{request_id}] 可视化结果已保存到: {visualized_path}")
            
            # 转换为Base64
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            buffer.seek(0)
            img_str = b64encode(buffer.getvalue()).decode('utf-8')
            
            visualize_time = time.time() - start_time
            logger.info(f"[{request_id}] 可视化完成，耗时: {visualize_time:.2f} 秒")
            return jsonify({"visualized_image": img_str, "save_path": visualized_path})
        else:
            # 如果没有检测结果，返回原图
            logger.info(f"[{request_id}] 无检测结果，返回原图")
            
            # 保存原图作为结果
            no_detection_path = os.path.join(timestamp_dir, f"no_detection_{request_id}.png")
            image_pil.save(no_detection_path)
            logger.info(f"[{request_id}] 无检测结果图像已保存到: {no_detection_path}")
            
            buffer = io.BytesIO()
            image_pil.save(buffer, format="PNG")
            buffer.seek(0)
            img_str = b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({"visualized_image": img_str, "message": "未检测到对象，返回原图", "save_path": no_detection_path})
            
    except Exception as e:
        logger.error(f"[{request_id}] 可视化处理失败: {str(e)}")
        
        # 保存错误信息
        error_info = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }
        error_json_path = os.path.join(timestamp_dir, f"visualize_error_{request_id}.json")
        with open(error_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        logger.info(f"[{request_id}] 可视化错误信息已保存到: {error_json_path}")
        
        return jsonify({"error": str(e)}), 500

def main():
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='LangSAM服务器')
        parser.add_argument('--gpu-device', type=int, default=None, help='指定使用的GPU设备ID，默认为0或环境变量GPU_DEVICE的值')
        parser.add_argument('--port', type=int, default=None, help='服务器端口号，默认为5000或环境变量PORT的值')
        args = parser.parse_args()
        
        logger.info("=== LangSAM服务器启动 ===")
        
        # 确保日志目录存在
        global log_dir
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"日志根目录: {os.path.abspath(log_dir)}")
        
        # 设置环境，传入GPU设备参数
        setup_environment(args.gpu_device)
        
        # 预加载模型
        model = get_model()
        
        # 获取端口设置
        port = args.port if args.port is not None else int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'  # 监听所有接口以允许局域网访问
        
        logger.info(f"服务器启动在 http://{host}:{port}")
        logger.info("API端点:")
        logger.info("  POST /predict - 图像分割和分析")
        logger.info("  POST /visualize - 结果可视化")
        logger.info("结果存储配置:")
        logger.info(f"  - JSON结果: ./log/时间戳/result_*.json")
        logger.info(f"  - 可视化结果: ./log/时间戳/visualized_*.png")
        logger.info(f"  - 服务器日志: ./log/lsamserver.log")
        
        # 启动服务器
        logger.info("开始监听请求...")
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，服务器正在关闭...")
    except Exception as e:
        logger.error(f"服务器意外关闭: {str(e)}")
    finally:
        logger.info("服务器已停止")

# 确保中文显示正常
app.logger = logger

if __name__ == "__main__":
    main()