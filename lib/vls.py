# vls.py - Vision Language System 封装文件
# 封装VLM和LLM模型的初始化和使用函数

import cv2
import json
import numpy as np
from lib.vlm import VLM, LLM
from lib.cloud_utils import PointCloudUtils

# 在文件顶部添加导入语句
from lib.solve import Target3DProcessor

from lib.log_utils import create_reader
from lib.lsamclient import getLSamResult
from lib.img_utils import ImageUtils
# 环境配置
REALDEVICE_ENV = False  # 是否为真机环境
if REALDEVICE_ENV:
    from lib.rosrgbd import ROSRGBD


def initModel():
    """
    初始化VLM和LLM模型
    
    Returns:
        tuple: (vlm, llm) - VLM和LLM模型实例
    """
    vlm = VLM()
    llm = LLM()
    return vlm, llm


def getCtrlCmd():
    """
    获取控制命令
    
    Returns:
        str: 控制命令字符串
    """
    try:
        with open('commend', 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"控制命令：{first_line}")
            return first_line
    except Exception as e:
        print(f"读取commend.txt失败: {e}")
        return ""


def getCameraData():
    """
    获取RGBD相机数据
    
    Returns:
        tuple: (rgb_image, depth_data) - RGB图像和深度数据
    """
    if REALDEVICE_ENV:
        # 真机环境下，使用ROSRGBD获取数据
        rgbd = ROSRGBD(log_level='error')
        color_image, depth_image, status = rgbd.getRGBD()
        rgbd.shutdown()
        return color_image, depth_image
    else:
        reader = create_reader('./oldlog')
        # 读取最新的日志数据
        latest_data = reader.read_log_data('20251103_105035')
        # 获取RGB图像
        rgb_image = latest_data['color_image']
        # 获取深度数据
        dep_data = latest_data['depth_data']

        return rgb_image, dep_data


def getPrompt(ctrl_cmd):
    """
    根据控制命令生成检测提示词和规划提示词
    
    Args:
        ctrl_cmd: 控制命令字符串
    
    Returns:
        tuple: (detect_prompt, plan_prompt) - 检测提示词和规划提示词
    """
    detect_prompt = f"""
    Analyze the drone control instruction: '{ctrl_cmd}'
    
    1. Extract the target object mentioned in the instruction
    2. Identify the specific object in the image if there are multiple similar objects
        (based on description like "on the left", "larger", "blue", etc.)
    3. Draw a bounding box around the entire target object
    
    Return the answer as a JSON array with the following format:
    [{{"label": "<target_object>","box_2d": [ymin, xmin, ymax, xmax]}}] where:
    - box_2d is [ymin, xmin, ymax, xmax] format normalized to 0-1000
    - All values in box_2d must be integers
    - label is the extracted target object name
    
    Never return any additional text, markdown formatting, or explanations.
    Only return the JSON array as specified.
"""

    plan_prompt = f"根据图像中的{ctrl_cmd}目标，生成一条规划路径"
    return detect_prompt, plan_prompt

def getDetectBBox(vlm, rgb_image, detect_prompt):
    """
    使用VLM根据自然语言控制指令进行目标检测，并将归一化坐标转换为实际像素坐标
    
    Args:
        vlm: VLM模型实例
        rgb_image: RGB格式的图像数据
        detect_prompt: 检测提示词
    
    Returns:
        list: 检测结果列表，包含实际像素坐标
    """
    # 调用VLM进行图像分析，获取归一化坐标的检测结果
    response_text = vlm.analyze_image(
        rgb_image=rgb_image,
        prompt=detect_prompt
    )
    
    try:
            normalized_result = vlm.parse_json_response(response_text)
    except json.JSONDecodeError:
        print("VLM未检测到目标对象")
        exit(0) 
    
    # 获取图像形状
    image_shape = rgb_image.shape[:2]  # (height, width)
    
    # 使用反归一化函数将坐标转换为实际像素坐标
    pixel_coordinates_result = vlm.denormalize_coordinates(
        normalized_result, 
        image_shape,
        verbose=False
    )
    
    return pixel_coordinates_result

def get3DTargetModel(rgb_image, detect_result, depth_data):
    """
    从RGB图像中剪切检测结果对应的区域，并构建指定格式的字典
    
    Args:
        rgb_image: RGB格式的图像数据
        detect_result: 检测结果列表，包含边界框信息
        depth_data: 深度图像数据
    
    Returns:
        list: 包含剪切区域信息的字典列表
    """
    
    processor = Target3DProcessor()
    return processor.process_targets(rgb_image, detect_result, depth_data)



def getPlan(llm, target3d, plan_prompt):
    pass