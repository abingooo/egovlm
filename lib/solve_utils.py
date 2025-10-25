import numpy as np


class DepthCalculator:
    """
    深度计算器类，用于处理不同类型检测结果的深度值计算
    
    支持两种检测类型：
    - point: 单点检测，直接获取检测点的深度值
    - box: 边界框检测，计算边界框中心的深度值
    """
    
    def __init__(self, depth_data, verbose=False, camera_params=None):
        """
        初始化深度计算器
        
        Args:
            depth_data (np.ndarray): 深度数据数组
            verbose (bool, optional): 是否打印详细信息. Defaults to False.
            camera_params: 相机内参字典，包含fx, fy, cx, cy
        """
        self.depth_data = depth_data
        self.verbose = verbose
        
        # 默认相机内参（可以根据实际相机参数修改）
        self.camera_params = camera_params or {
            'fx': 383.19929174573906,  # 焦距x
            'fy': 384.76715878730715,  # 焦距y
            'cx': 317.944484051631,  # 光心x
            'cy': 231.71115593384292   # 光心y
        }
        
        # 获取深度数据形状
        self.depth_height, self.depth_width = depth_data.shape[:2]
    
    def get_depth_from_detections(self, detections, rgb_shape=None):
        """
        从检测结果中计算深度值
        
        Args:
            detections (list): 检测结果列表，包含point或box类型的检测结果
            rgb_shape (tuple, optional): RGB图像形状，用于调试信息显示
            
        Returns:
            list: 每个检测结果对应的深度值列表
        """
        if not detections:
            if self.verbose:
                print("没有有效的检测结果")
            return []
        
        # 打印调试信息
        if self.verbose and rgb_shape:
            print(f"RGB图像形状: {rgb_shape}")
            print(f"深度数据形状: {self.depth_data.shape}")
        
        depth_values = []
        # 对每个检测结果单独计算深度
        for detection in detections:
            if 'point' in detection:
                depth = self._get_point_depth(detection)
                depth_values.append(depth)
            elif 'box' in detection or 'box_2d' in detection:
                # 支持box和box_2d两种键名
                box_key = 'box' if 'box' in detection else 'box_2d'
                detection_copy = detection.copy()
                # 创建临时检测字典，使用box键名
                detection_copy['box'] = detection[box_key]
                depth = self._get_box_center_depth(detection_copy)
                depth_values.append(depth)
            else:
                if self.verbose:
                    print(f"未知的检测结果类型: {detection}")
                depth_values.append(None)
        
        return depth_values
    
    def _get_point_depth(self, detection):
        """
        获取点检测结果的深度值
        
        Args:
            detection (dict): 包含point信息的检测结果
            
        Returns:
            float or None: 点的深度值
        """
        y, x = detection['point']
        
        if self.verbose:
            print(f"检测点坐标 (y,x): ({y}, {x})"),
        
        # 确保坐标在有效范围内
        x = max(0, min(x, self.depth_width - 1))
        y = max(0, min(y, self.depth_height - 1))
        
        # 访问深度数据
        depth_value = self.depth_data[y, x]
        
        if self.verbose:
            print(f"检测点的深度值: {depth_value}")
        
        return depth_value
    
    def _get_box_center_depth(self, detection):
        """
        获取边界框中心的深度值
        
        Args:
            detection (dict): 包含box信息的检测结果
            
        Returns:
            float or None: 边界框中心的深度值
        """
        # 处理box类型：计算矩形框中心坐标
        ymin, xmin, ymax, xmax = detection['box']
        
        if self.verbose:
            print(f"检测框坐标 [ymin, xmin, ymax, xmax]: [{ymin}, {xmin}, {ymax}, {xmax}]")
        
        # 计算框中心坐标
        y_center = int((ymin + ymax) / 2)
        x_center = int((xmin + xmax) / 2)
        
        if self.verbose:
            print(f"检测框中心坐标 (y,x): ({y_center}, {x_center})")
        
        # 确保中心坐标在有效范围内
        x_center = max(0, min(x_center, self.depth_width - 1))
        y_center = max(0, min(y_center, self.depth_height - 1))
        
        # 访问中心处的深度数据
        depth_value = self.depth_data[y_center, x_center]
        
        if self.verbose:
            print(f"检测框中心的深度值: {depth_value}")
        
        return depth_value
    
    def pixel_to_3d(self, y, x, depth, convert_coordinate_system=False):
        """
        将像素坐标和深度值转换为3D坐标
        
        Args:
            y: 像素y坐标
            x: 像素x坐标
            depth: 深度值（毫米）
            convert_coordinate_system: 是否转换坐标系，True时使用z向上、x向前、y向左的坐标系
        
        Returns:
            tuple: (X, Y, Z) 3D坐标（米）
        """
        # 如果深度无效，返回None
        if depth is None or depth <= 0:
            if self.verbose:
                print(f"无效的深度值: {depth}")
            return None
        
        # 转换深度为米
        Z = depth * 0.001
        
        # 获取相机参数
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']
        
        # 计算相机坐标系下的3D坐标
        # 相机坐标系：X向右，Y向下，Z向前
        camera_X = (x - cx) * Z / fx
        camera_Y = (y - cy) * Z / fy
        camera_Z = Z
        
        if not convert_coordinate_system:
            # 返回相机坐标系
            if self.verbose:
                print(f"像素坐标 ({y}, {x}) 转换为相机坐标系: ({camera_X:.3f}, {camera_Y:.3f}, {camera_Z:.3f}) 米")
            return (camera_X, camera_Y, camera_Z)
        else:
            # 转换到指定坐标系：Z向上，X向前，Y向左
            # 变换关系：
            # 新X = 相机Z
            # 新Y = -相机X
            # 新Z = -相机Y
            new_X = camera_Z
            new_Y = -camera_X
            new_Z = -camera_Y
            
            if self.verbose:
                print(f"像素坐标 ({y}, {x}) 转换为指定坐标系: ({new_X:.3f}, {new_Y:.3f}, {new_Z:.3f}) 米")
            return (new_X, new_Y, new_Z)
    
    def detection_to_3d(self, detection, depth=None, convert_coordinate_system=False):
        """
        将检测结果转换为3D坐标
        
        Args:
            detection: 单个检测结果字典
            depth: 可选的深度值，如果不提供则自动计算
            convert_coordinate_system: 是否转换坐标系，True时使用z向上、x向前、y向左的坐标系
        
        Returns:
            tuple: (X, Y, Z) 3D坐标（米），或None
        """
        # 如果没有提供深度值，则计算
        if depth is None:
            if 'point' in detection:
                depth = self._get_point_depth(detection)
            elif 'box' in detection or 'box_2d' in detection:
                # 支持box和box_2d两种键名
                box_key = 'box' if 'box' in detection else 'box_2d'
                detection_copy = detection.copy()
                # 创建临时检测字典，使用box键名
                detection_copy['box'] = detection[box_key]
                depth = self._get_box_center_depth(detection_copy)
            else:
                if self.verbose:
                    print(f"未知的检测结果类型: {detection}")
                return None
        
        # 根据检测类型获取像素坐标
        if 'point' in detection:
            y, x = detection['point']
        elif 'box' in detection or 'box_2d' in detection:
            # 支持box和box_2d两种键名
            box_key = 'box' if 'box' in detection else 'box_2d'
            ymin, xmin, ymax, xmax = detection[box_key]
            y = (ymin + ymax) / 2
            x = (xmin + xmax) / 2
        else:
            return None
        
        # 转换为3D坐标，传递坐标系转换参数
        return self.pixel_to_3d(y, x, depth, convert_coordinate_system)
    
    def detections_to_3d_positions(self, detections, depth_values=None, convert_coordinate_system=False):
        """
        将检测结果列表转换为3D坐标列表
        
        Args:
            detections: 检测结果列表
            depth_values: 可选的深度值列表，如果不提供则自动计算
            convert_coordinate_system: 是否转换坐标系，True时使用z向上、x向前、y向左的坐标系
        
        Returns:
            list: 3D坐标列表 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
        """
        if not detections:
            return []
        
        # 如果没有提供深度值列表，则计算
        if depth_values is None:
            depth_values = self.get_depth_from_detections(detections)
        
        # 计算每个检测结果的3D坐标
        positions_3d = []
        for i, detection in enumerate(detections):
            depth = depth_values[i] if i < len(depth_values) else None
            position_3d = self.detection_to_3d(detection, depth, convert_coordinate_system)
            positions_3d.append(position_3d)
        
        return positions_3d


# 便捷函数 - 支持直接通过坐标获取深度值
def calculate_depth(depth_data, y, x):
    """
    便捷函数：直接通过坐标获取深度值
    
    Args:
        depth_data (np.ndarray): 深度数据数组
        y (int): y坐标
        x (int): x坐标
        
    Returns:
        float: 深度值
    """
    # 获取深度数据形状
    depth_height, depth_width = depth_data.shape[:2]
    # 确保坐标在有效范围内
    x = max(0, min(x, depth_width - 1))
    y = max(0, min(y, depth_height - 1))
    # 返回深度值
    return depth_data[y, x]


# 通过检测结果获取深度值
def calculate_depth_from_detections(detections, depth_data, rgb_shape=None, verbose=False):
    """
    便捷函数：计算检测结果的深度值
    
    Args:
        detections (list): 检测结果列表
        depth_data (np.ndarray): 深度数据
        rgb_shape (tuple, optional): RGB图像形状
        verbose (bool, optional): 是否打印详细信息
        
    Returns:
        list: 每个检测结果对应的深度值列表
    """
    calculator = DepthCalculator(depth_data, verbose=verbose)
    return calculator.get_depth_from_detections(detections, rgb_shape)


# 通过检测结果获取3D位置
def calculate_3d_position(detections, depth_data, rgb_shape=None, verbose=False, camera_params=None, convert_coordinate_system=False):
    """
    便捷函数：从检测结果和深度数据计算3D位置
    
    Args:
        detections (list): 检测结果列表
        depth_data (np.ndarray): 深度数据
        rgb_shape (tuple, optional): RGB图像形状
        verbose (bool, optional): 是否打印详细信息
        camera_params: 相机内参字典
        convert_coordinate_system: 是否转换坐标系，True时使用z向上、x向前、y向左的坐标系
    
    Returns:
        list: 每个检测结果对应的3D坐标列表 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
    """
    calculator = DepthCalculator(depth_data, verbose=verbose, camera_params=camera_params)
    return calculator.detections_to_3d_positions(detections, convert_coordinate_system=convert_coordinate_system)