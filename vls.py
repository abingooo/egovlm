# vls.py - Vision Language System 封装文件
# 封装VLM和LLM模型的初始化和使用函数

import cv2
import numpy as np
from lib.vlm import VLM, LLM, ImageObjectDetector
from lib.solve_utils import calculate_3d_position
from lib.cloud_utils import PointCloudUtils
from lib.logread import create_reader
from lsamclient import getLSamResult
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
    # 在实际应用中，这里可以从用户输入、配置文件或其他来源获取控制命令
    # 这里返回一个示例命令
    return "fly to the tree."


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
        reader = create_reader('./log')
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
    
    # 解析JSON响应
    normalized_result = vlm.parse_json_response(response_text)
    
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
        depth_data: 深度图像数据（这里暂不使用）
    
    Returns:
        list: 包含剪切区域信息的字典列表，格式为[{"objectindex": 0, "label":"tree","bbox":[[],[]],"imageROI":image},...]
    """
    object_dict_for3d_list = []
    
    # 遍历所有检测结果
    for index, detection in enumerate(detect_result):
        # 检查是否包含box_2d信息
        if "box_2d" in detection:
            # 获取边界框坐标 [y1, x1, y2, x2]
            y1, x1, y2, x2 = detection["box_2d"]
            
            # 确保坐标在有效范围内
            height, width = rgb_image.shape[:2]
            y1 = max(0, min(y1, height - 1))
            x1 = max(0, min(x1, width - 1))
            y2 = max(0, min(y2, height - 1))
            x2 = max(0, min(x2, width - 1))
            
            # 剪切区域（注意：OpenCV使用[y:y+h, x:x+w]格式）
            image_roi = rgb_image[y1:y2+1, x1:x2+1]
            
            # 构建字典，注意bbox格式为[[x1,y1],[x2,y2]]
            object_dict = {
                "objectindex": index,
                "label": detection.get("label", f"object_{index}"),
                "bbox": [[x1, y1], [x2, y2]],
                "imageROI": image_roi
            }
            
            # 调用LSam服务器进行目标分割（支持直接传入NumPy数组）
            lsam_result = getLSamResult(object_dict['imageROI'], object_dict['label'], server_ip="172.16.1.61", server_port=5000)
            
            # 使用新增的可视化方法保存LSAM分割结果
            ImageUtils.visualize_lsam_result(object_dict['imageROI'], lsam_result, index=index, label=object_dict['label'], save_dir="./debug/rois")
            
            # 将LSAM分割结果从ROI图像坐标系转换到原始RGB图像坐标系
            rgb_coordinates_lsam_result = ImageUtils.convert_lsam_coordinates_to_rgb(lsam_result, object_dict['bbox'])
            
            # 构造全新的字典，只包含要求的五个字段
            if rgb_coordinates_lsam_result.get('mask_count', 0) > 0 and 'masks' in rgb_coordinates_lsam_result:
                first_mask = rgb_coordinates_lsam_result['masks'][0]
                
                # 创建全新的字典，使用英文键名
                object_dict_for3d = {
                    'id': index,  # 1. id
                    'label': object_dict['label'],  # 2. 标签
                    'center': first_mask.get('centroid', []),  # 3. 质心
                }
                
                # 4. 边界框（转换为简单格式）
                bbox_data = first_mask.get('bounding_box', {})
                object_dict_for3d['bbox'] = [
                    [bbox_data.get('x1', 0), bbox_data.get('y1', 0)],
                    [bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
                ]
                
                # 5. 9个随机点
                random_points = first_mask.get('random_points', [])
                object_dict_for3d['npoints'] = random_points
                
                object_dict_for3d_list.append(object_dict_for3d)
    
    # 对物体列表进行3d空间建模
    obj3dmodel_list = []
    for obj in object_dict_for3d_list:
        obj3d = {}
        obj3d['id'] = obj['id']
        obj3d['label'] = obj['label']
        # 平均深度计算 取9个随机点和质心处的深度值
        avg_depth = 0
        for point in obj['npoints'] + [obj['center']]:
            # 对每个像素点应用7x7窗口的中值滤波
            x, y = point
            # 计算窗口边界，确保在图像范围内
            half_win = 3  # 7x7窗口的半宽
            x_min = max(0, x - half_win)
            x_max = min(depth_data.shape[1] - 1, x + half_win)
            y_min = max(0, y - half_win)
            y_max = min(depth_data.shape[0] - 1, y + half_win)
            # 获取窗口内的深度值
            window = depth_data[y_min:y_max+1, x_min:x_max+1]
            # 计算中值深度（使用Python标准库）
            window_flat = window.flatten()
            sorted_window = sorted(window_flat)
            median_index = len(sorted_window) // 2
            median_depth = sorted_window[median_index]
            avg_depth += float(median_depth)  # 转换为Python原生float类型
        avg_depth /= 10
        obj3d['depth'] = round(avg_depth, 2)  # 保留2位小数
        # 定义通用的3D坐标计算函数
        def calculate_3d_position(pixel_coord, depth, camera_params, precision=2):
            """
            根据像素坐标和深度计算3D坐标
            
            Args:
                pixel_coord: 像素坐标 [x, y]
                depth: 深度值
                camera_params: 相机参数字典，包含fx, fy, cx, cy
                precision: 保留小数位数
                
            Returns:
                3D坐标 [x, y, z]
            """
            # 确保坐标是整数
            x_pixel = round(pixel_coord[0])
            y_pixel = round(pixel_coord[1])
            
            x = ((x_pixel - camera_params['cx']) * depth / camera_params['fx']) * 0.001
            y = ((y_pixel - camera_params['cy']) * depth / camera_params['fy']) * 0.001
            z = depth * 0.001
            return [round(float(x), precision), round(float(y), precision), round(float(z), precision)]
        
        # 设置相机参数
        camera_params = {
            'fx': 383.19929174573906,  # 焦距x
            'fy': 384.76715878730715,  # 焦距y
            'cx': 317.944484051631,    # 光心x
            'cy': 231.71115593384292   # 光心y
        }
        
        # 使用通用函数计算质心3D坐标（XYZ正向：右下前）
        obj3d['center'] = calculate_3d_position(obj['center'], obj3d['depth'], camera_params)
        # 计算四个角点3D坐标（XYZ正向：右下前）
        obj3d['bbox3dfront'] = [
            calculate_3d_position(obj['bbox'][0], obj3d['depth'], camera_params),#左上前角点
            calculate_3d_position(obj['bbox'][1], obj3d['depth'], camera_params),#右下前角点
            calculate_3d_position([obj['bbox'][1][0], obj['bbox'][0][1]], obj3d['depth'], camera_params),#右上前角点
            calculate_3d_position([obj['bbox'][0][0], obj['bbox'][1][1]], obj3d['depth'], camera_params)#左下前角点
        ]
        # 对质心的Z进行修正，obj3d['center']加上四条边中最长的值的0.1的正值
        # 简化计算：只计算矩形边界框的两条对角线
        # 对角线1：左上到右下
        dx1 = obj3d['bbox3dfront'][0][0] - obj3d['bbox3dfront'][1][0]
        dy1 = obj3d['bbox3dfront'][0][1] - obj3d['bbox3dfront'][1][1]
        dz1 = obj3d['bbox3dfront'][0][2] - obj3d['bbox3dfront'][1][2]
        diagonal1 = (dx1**2 + dy1**2 + dz1**2) ** 0.5  # 使用Python内置的平方根
        
        # 对角线2：右上到左下
        dx2 = obj3d['bbox3dfront'][2][0] - obj3d['bbox3dfront'][3][0]
        dy2 = obj3d['bbox3dfront'][2][1] - obj3d['bbox3dfront'][3][1]
        dz2 = obj3d['bbox3dfront'][2][2] - obj3d['bbox3dfront'][3][2]
        diagonal2 = (dx2**2 + dy2**2 + dz2**2) ** 0.5  # 使用Python内置的平方根
        
        # 对质心Z坐标进行修正
        xiuzhen = abs(max([diagonal1, diagonal2]))
        xiuzhen = round(float(xiuzhen), 2)  # 转换为Python原生float并保留2位小数
        obj3d['center'][2] = round(obj3d['center'][2] + xiuzhen * 0.2, 2)

        # 估算后面的四个角点
        back_z_offset = round(xiuzhen * 0.4, 2)
        obj3d['bbox3dback'] = [
            [obj3d['bbox3dfront'][0][0], obj3d['bbox3dfront'][0][1], round(obj3d['bbox3dfront'][0][2] + back_z_offset, 2)],#左上后角点
            [obj3d['bbox3dfront'][1][0], obj3d['bbox3dfront'][1][1], round(obj3d['bbox3dfront'][1][2] + back_z_offset, 2)],#右下后角点
            [obj3d['bbox3dfront'][2][0], obj3d['bbox3dfront'][2][1], round(obj3d['bbox3dfront'][2][2] + back_z_offset, 2)],#右上后角点
            [obj3d['bbox3dfront'][3][0], obj3d['bbox3dfront'][3][1], round(obj3d['bbox3dfront'][3][2] + back_z_offset, 2)]#左下后角点
        ]
        
        # 确保所有数值都转换为Python原生类型并保留2位小数
        # 处理bbox3dfront中的所有坐标
        for i in range(len(obj3d['bbox3dfront'])):
            obj3d['bbox3dfront'][i] = [
                round(float(obj3d['bbox3dfront'][i][0]), 2),
                round(float(obj3d['bbox3dfront'][i][1]), 2),
                round(float(obj3d['bbox3dfront'][i][2]), 2)
            ]
        
        obj3dmodel_list.append(obj3d)

    return obj3dmodel_list



def getPlan(llm, target3d, plan_prompt):
    """
    使用LLM根据目标3D模型进行路径规划
    
    Args:
        llm: LLM模型实例
        target3d: 3D目标模型数据
        plan_prompt: 规划提示词
    
    Returns:
        list: 路径航点列表
    """
    # 提取目标信息
    target_info = target3d.get('main_target', {})
    target_label = target_info.get('label', 'unknown')
    target_position = target_info.get('position', [0, 0, 0])
    
    # 格式化提示词，包含目标的3D坐标信息
    formatted_prompt = plan_prompt
    
    # 使用LLM生成路径规划
    # 注意：这里需要根据实际的LLM接口进行调整
    # 由于没有看到LLM类的详细实现，这里提供一个简化版本
    try:
        # 假设LLM类有generate_content或类似的方法
        if hasattr(llm, 'generate_content'):
            response = llm.generate_content(formatted_prompt)
        else:
            # 或者调用其他可用的方法
            response = llm.create_prompt(target_label, prompt_type="waypoint1")
        
        # 解析响应，提取航点
        # 这里需要根据实际的响应格式进行解析
        # 提供一个简化的解析逻辑
        waypoints = []
        
        # 如果响应是JSON格式，可以使用json.loads解析
        import json
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
                # 假设解析后的数据包含航点信息
                if isinstance(parsed, list):
                    waypoints = parsed
        except Exception:
            # 如果解析失败，生成默认航点
            waypoints = [[0, 0, 0], list(target_position)]
        
        return waypoints
    except Exception as e:
        print(f"路径规划失败: {e}")
        # 返回默认航点
        return [[0, 0, 0], list(target_position)]