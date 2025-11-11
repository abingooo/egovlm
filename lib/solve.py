import numpy as np
from lib.lsamclient import getLSamResult
from lib.img_utils import ImageUtils

class TargetExtractor:
    """
    从RGB图像中提取目标区域的类
    """
    
    @staticmethod
    def extract_target_region(rgb_image, detection):
        """
        从RGB图像中提取目标区域
        
        Args:
            rgb_image: RGB格式的图像数据
            detection: 单个检测结果，包含边界框信息
            
        Returns:
            dict: 包含目标区域信息的字典
        """
        # 检查是否包含box_2d信息
        if "box_2d" not in detection:
            return None
        
        # 对边界框坐标进行外扩，增大sam分割识别视野
        detection["box_2d"][0] = int(detection["box_2d"][0] * 0.8)
        detection["box_2d"][1] = int(detection["box_2d"][1] * 0.8)
        detection["box_2d"][2] = int(detection["box_2d"][2] * 1.2)
        detection["box_2d"][3] = int(detection["box_2d"][3] * 1.2)
        
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
        return {
            "label": detection.get("label", "unknown"),
            "bbox": [[x1, y1], [x2, y2]],
            "imageROI": image_roi
        }

class LSamProcessor:
    """
    处理LSam服务器分割结果的类
    """
    
    @staticmethod
    def process_lsam_result(target_region, index, server_ip="172.16.1.61", server_port=5000):
        """
        处理LSam服务器的分割结果
        
        Args:
            target_region: 目标区域信息字典
            index: 目标索引
            server_ip: LSam服务器IP地址
            server_port: LSam服务器端口
            
        Returns:
            dict: 转换到原始RGB图像坐标系的LSam分割结果
        """
        # 调用LSam服务器进行目标分割（支持直接传入NumPy数组）
        lsam_result = getLSamResult(
            target_region['imageROI'], 
            target_region['label'], 
            server_ip=server_ip, 
            server_port=server_port
        )

        # 只选取最大面积目标
        lsam_result['masks'] = [max(lsam_result['masks'], key=lambda x: x['area'])]
        lsam_result['mask_count'] = 1
        
        # 使用新增的可视化方法保存LSAM分割结果
        ImageUtils.visualize_lsam_result(
            target_region['imageROI'], 
            lsam_result, 
            index=index, 
            label=target_region['label'], 
            save_dir="./log/roi"
        )
        
        # 将LSAM分割结果从ROI图像坐标系转换到原始RGB图像坐标系
        return ImageUtils.convert_lsam_coordinates_to_rgb(lsam_result, target_region['bbox'])

class Target3DModeler:
    """
    构建3D目标模型的类
    """
    
    def __init__(self):
        # 设置相机参数
        self.camera_params = {
            'fx': 383.19929174573906,  # 焦距x
            'fy': 384.76715878730715,  # 焦距y
            'cx': 317.944484051631,    # 光心x
            'cy': 231.71115593384292   # 光心y
        }
    
    def calculate_3d_position(self, pixel_coord, depth, precision=2):
        """
        根据像素坐标和深度计算3D坐标
        
        Args:
            pixel_coord: 像素坐标 [x, y]
            depth: 深度值
            precision: 保留小数位数
            
        Returns:
            3D坐标 [x, y, z]
        """
        # 确保坐标是整数
        x_pixel = round(pixel_coord[0])
        y_pixel = round(pixel_coord[1])
        
        x = ((x_pixel - self.camera_params['cx']) * depth / self.camera_params['fx']) * 0.001
        y = ((y_pixel - self.camera_params['cy']) * depth / self.camera_params['fy']) * 0.001
        z = depth * 0.001
        return [round(float(x), precision), round(float(y), precision), round(float(z), precision)]
    
    def calculate_average_depth(self, obj_data, depth_data):
        """
        计算目标的平均深度
        
        Args:
            obj_data: 目标数据字典
            depth_data: 深度图像数据
            
        Returns:
            float: 平均深度值
        """
        avg_depth = 0
        for point in obj_data['npoints'] + [obj_data['center']]:
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
        return round(avg_depth, 2)  # 保留2位小数
    
    def create_3d_model(self, rgb_coordinates_lsam_result, depth_data, index):
        """
        创建3D目标模型
        
        Args:
            rgb_coordinates_lsam_result: 转换后的LSam结果
            depth_data: 深度图像数据
            index: 目标索引
            
        Returns:
            dict: 3D目标模型字典
        """
        if rgb_coordinates_lsam_result.get('mask_count', 0) <= 0 or 'masks' not in rgb_coordinates_lsam_result:
            return None
        
        first_mask = rgb_coordinates_lsam_result['masks'][0]

        # 创建对象数据字典
        obj_data = {
            'id': index,
            'label': rgb_coordinates_lsam_result.get('text_prompt', f'object_{index}'),
            'center': first_mask.get('centroid', []),
        }
        
        # 边界框（转换为简单格式）
        bbox_data = first_mask.get('bounding_box', {})
        obj_data['bbox'] = [
            [bbox_data.get('x1', 0), bbox_data.get('y1', 0)],
            [bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
        ]
        
        # 9个随机点
        random_points = first_mask.get('random_points', [])
        obj_data['npoints'] = random_points
        
        # 创建3D模型
        obj3d = {
            'id': obj_data['id'],
            'label': obj_data['label']
        }
        
        # 计算平均深度
        obj3d['depth'] = self.calculate_average_depth(obj_data, depth_data)
        
        # 使用通用函数计算质心3D坐标（XYZ正向：右下前）
        obj3d['center'] = self.calculate_3d_position(obj_data['center'], obj3d['depth'])
        
        #  ============立方体建模计算=============================
        # 计算四个角点3D坐标（XYZ正向：右下前）
        obj3d['bbox3dfront'] = [
            self.calculate_3d_position(obj_data['bbox'][0], obj3d['depth']),  # 左上前角点
            self.calculate_3d_position(obj_data['bbox'][1], obj3d['depth']),  # 右下前角点
            self.calculate_3d_position([obj_data['bbox'][1][0], obj_data['bbox'][0][1]], obj3d['depth']),  # 右上前角点
            self.calculate_3d_position([obj_data['bbox'][0][0], obj_data['bbox'][1][1]], obj3d['depth'])  # 左下前角点
        ]

        # 对质心的Z进行修正
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
            [obj3d['bbox3dfront'][0][0], obj3d['bbox3dfront'][0][1], round(obj3d['bbox3dfront'][0][2] + back_z_offset, 2)],  # 左上后角点
            [obj3d['bbox3dfront'][1][0], obj3d['bbox3dfront'][1][1], round(obj3d['bbox3dfront'][1][2] + back_z_offset, 2)],  # 右下后角点
            [obj3d['bbox3dfront'][2][0], obj3d['bbox3dfront'][2][1], round(obj3d['bbox3dfront'][2][2] + back_z_offset, 2)],  # 右上后角点
            [obj3d['bbox3dfront'][3][0], obj3d['bbox3dfront'][3][1], round(obj3d['bbox3dfront'][3][2] + back_z_offset, 2)]  # 左下后角点
        ]
        #  ============球体建模计算=============================
        # 估算物体中心
        center_x = sum(pt[0] for pt in obj3d['bbox3dfront']) / 4
        center_y = sum(pt[1] for pt in obj3d['bbox3dfront']) / 4
        center_z = sum(pt[2] for pt in obj3d['bbox3dfront']) / 4
        front3d_center = [round(center_x, 2), round(center_y, 2), round(center_z, 2)]
        obj3d['circle_center'] = front3d_center
        obj3d['circle_center'][2] = round(obj3d['circle_center'][2] + xiuzhen * 0.35, 2)
        # 计算最大半径
        obj3d['radius'] = round(xiuzhen * 0.55, 2)
        
        # 确保所有数值都转换为Python原生类型并保留2位小数
        # 处理bbox3dfront中的所有坐标
        for i in range(len(obj3d['bbox3dfront'])):
            obj3d['bbox3dfront'][i] = [
                round(float(obj3d['bbox3dfront'][i][0]), 2),
                round(float(obj3d['bbox3dfront'][i][1]), 2),
                round(float(obj3d['bbox3dfront'][i][2]), 2)
            ]
        
        return obj3d


class Target3DProcessor:
    """
    整合所有功能的3D目标处理器类
    """
    
    def __init__(self):
        self.target_extractor = TargetExtractor()
        self.lsam_processor = LSamProcessor()
        self.target_3d_modeler = Target3DModeler()
    
    def process_targets(self, rgb_image, detect_result, depth_data):
        """
        处理检测结果，构建3D目标模型
        
        Args:
            rgb_image: RGB格式的图像数据
            detect_result: 检测结果列表，包含边界框信息
            depth_data: 深度图像数据
            
        Returns:
            list: 3D目标模型列表
        """
        object_dict_for3d_list = []
        
        # 遍历所有检测结果
        for index, detection in enumerate(detect_result):
            # 提取目标区域
            target_region = self.target_extractor.extract_target_region(rgb_image, detection)
            if target_region is None:
                continue
            
            # 处理LSam结果
            rgb_coordinates_lsam_result = self.lsam_processor.process_lsam_result(
                target_region, index
            )
            # import json
            # print(json.dumps(rgb_coordinates_lsam_result, indent=2, ensure_ascii=False))
            # 可视化lsam结果
            ImageUtils.visualize_target2d_results(rgb_image, rgb_coordinates_lsam_result)
            
            # 构建3D模型
            obj3d = self.target_3d_modeler.create_3d_model(
                rgb_coordinates_lsam_result, depth_data, index
            )
            if obj3d is not None:
                object_dict_for3d_list.append(obj3d)
        
        
        return object_dict_for3d_list
