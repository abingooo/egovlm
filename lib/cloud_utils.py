import numpy as np
import math
import os

class PointCloudUtils:
    """
    点云处理工具类，提供点云读写、处理和可视化相关功能
    """
    
    # 预定义颜色常量
    RED = [255, 0, 0]      # 红色：用于起始点的球体
    BLUE = [0, 0, 255]     # 蓝色：用于所有箭头
    GREEN = [0, 255, 0]    # 绿色：用于最后一个点
    PURPLE = [128, 0, 128] # 紫色：用于只有一个点的情况
    
    def __init__(self):
        """初始化点云工具类"""
        pass
    
    def read_ply_file(self, ply_file_path):
        """
        读取PLY文件，提取点云数据
        
        Args:
            ply_file_path: PLY文件路径
        
        Returns:
            tuple: (points, colors) 点云坐标和颜色数据
        """
        points = []
        colors = []
        
        try:
            with open(ply_file_path, 'r') as f:
                lines = f.readlines()
            
            # 跳过头部信息，找到数据开始位置
            header_end_index = 0
            vertex_count = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    header_end_index = i + 1
                    break
            
            # 读取点云数据
            for line in lines[header_end_index:header_end_index + vertex_count]:
                data = list(map(float, line.strip().split()))
                if len(data) >= 3:
                    points.append(data[:3])  # x, y, z
                    if len(data) >= 6:
                        colors.append(data[3:6])  # r, g, b
                    else:
                        colors.append([0, 0, 0])
            
            return np.array(points), np.array(colors)
        except Exception as e:
            print(f"读取PLY文件时出错: {e}")
            return np.array([]), np.array([])
    
    def add_annotation_points(self, input_ply_path, output_ply_path, annotation_points, radius=0.2, points_per_arrow=1000):
        """
        添加标注点到PLY点云文件（使用箭头显示顺序）
        
        参数:
            input_ply_path: 输入PLY文件路径
            output_ply_path: 输出PLY文件路径
            annotation_points: 标注点的3D坐标列表，格式为 [[x1, y1, z1], [x2, y2, z2], ...]
            annotation_color: 标注点颜色，默认红色
            radius: 标注点大小，默认0.2
            points_per_arrow: 每个箭头生成的点数，默认1000
        
        返回:
            bool: 是否成功添加标注点
        """
        output_ply_path = self.get_next_filename(output_ply_path)
        try:
            # 读取原始点云
            original_points, original_colors = self.read_ply_file(input_ply_path)
            if len(original_points) == 0:
                return False
            
            # 将标注点扩展为球体/箭头，使其在点云中更明显
            all_points = original_points.tolist()
            all_colors = original_colors.tolist()
            
            # 添加所有标注点，使用箭头显示顺序
            for i, point in enumerate(annotation_points):
                if len(annotation_points) == 1:
                    # 只有一个点的情况：显示紫色球体
                    sphere_points, sphere_colors = self._generate_sphere_points(
                        point, radius, points_per_arrow, self.PURPLE
                    )
                    all_points.extend(sphere_points)
                    all_colors.extend(sphere_colors)
                elif i == 0:
                    # 起始点：红色球体 + 蓝色箭头
                    # 1. 添加红色球体
                    sphere_points, sphere_colors = self._generate_sphere_points(
                        point, radius, points_per_arrow, self.RED
                    )
                    all_points.extend(sphere_points)
                    all_colors.extend(sphere_colors)
                    
                    # 2. 添加指向第二个点的蓝色箭头
                    next_point = annotation_points[i + 1]
                    arrow_points, arrow_colors = self._generate_arrow_points(
                        point, next_point, radius, points_per_arrow, self.BLUE
                    )
                    all_points.extend(arrow_points)
                    all_colors.extend(arrow_colors)
                elif i < len(annotation_points) - 1:
                    # 中间点：蓝色箭头（不显示球体）
                    next_point = annotation_points[i + 1]
                    arrow_points, arrow_colors = self._generate_arrow_points(
                        point, next_point, radius, points_per_arrow, self.BLUE
                    )
                    all_points.extend(arrow_points)
                    all_colors.extend(arrow_colors)
                else:
                    # 最后一个点：绿色球体
                    sphere_points, sphere_colors = self._generate_sphere_points(
                        point, radius, points_per_arrow, self.GREEN
                    )
                    all_points.extend(sphere_points)
                    all_colors.extend(sphere_colors)
            
            # 构建新的PLY文件
            total_points = len(all_points)
            
            # 创建PLY文件头部
            header = [
                'ply',
                'format ascii 1.0',
                f'element vertex {total_points}',
                'property float x',
                'property float y',
                'property float z',
                'property uchar red',
                'property uchar green',
                'property uchar blue',
                'end_header'
            ]
            
            # 写入新的PLY文件
            with open(output_ply_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(header) + '\n')
                
                for i in range(total_points):
                    x, y, z = all_points[i]
                    r = max(0, min(255, int(all_colors[i][0])))
                    g = max(0, min(255, int(all_colors[i][1])))
                    b = max(0, min(255, int(all_colors[i][2])))
                    f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
            
            print(f"成功添加 {len(annotation_points)} 个标注点到 {output_ply_path}，")
            print(f"使用箭头显示顺序，箭头指向后续点")
            return True
        
        except Exception as e:
            print(f"添加标注点时出错: {e}")
            return False
    
    def save_point_cloud_to_ply(self, points_3d, colors, ply_file_path):
        """
        将点云和颜色数据保存为PLY格式文件
        
        参数:
            points_3d: 3D点云数据，格式为 [[x1, y1, z1], [x2, y2, z2], ...]
            colors: 对应点的颜色数据，格式为 [[r1, g1, b1], [r2, g2, b2], ...]
            ply_file_path: 输出PLY文件路径
        
        返回:
            bool: 是否成功保存
        """
        try:
            # 确保points_3d不为空
            if len(points_3d) == 0:
                print("错误: 没有点云数据可保存")
                return False
            
            # 创建PLY文件头部
            header = [
                'ply',
                'format ascii 1.0',
                f'element vertex {len(points_3d)}',
                'property float x',
                'property float y',
                'property float z',
                'property uchar red',
                'property uchar green',
                'property uchar blue',
                'end_header'
            ]
            
            # 将点云和颜色数据组合，并确保颜色值在有效范围内
            point_cloud_data = []
            for i in range(len(points_3d)):
                x, y, z = points_3d[i]
                # 确保颜色值在0-255范围内
                if i < len(colors):
                    r = max(0, min(255, int(colors[i, 0])))
                    g = max(0, min(255, int(colors[i, 1])))
                    b = max(0, min(255, int(colors[i, 2])))
                else:
                    r, g, b = 0, 0, 0
                # 格式化为PLY文件要求的格式
                point_cloud_data.append(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}')
            
            # 写入PLY文件，使用二进制模式的文本写入以确保格式正确
            with open(ply_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(header) + '\n')
                # 逐行写入以避免大文件时的内存问题
                for line in point_cloud_data:
                    f.write(line + '\n')
            
            print(f'点云已成功保存到: {ply_file_path}')
            print(f'保存的点云包含 {len(points_3d)} 个点')
            return True
        except Exception as e:
            print(f"保存点云到PLY文件时出错: {e}")
            return False
    
    def depth_to_point_cloud(self, depth_data, rgb_image, fx, fy, cx, cy, depth_scale=0.001):
        """
        从深度图和RGB图像生成3D点云
        
        参数:
            depth_data: 深度图像数据
            rgb_image: RGB彩色图像
            fx, fy: 相机的焦距
            cx, cy: 相机的主点坐标
            depth_scale: 深度值缩放因子，将深度值转换为米
        
        返回:
            tuple: (points_3d, colors) 生成的3D点云和对应颜色
        """
        try:
            # 过滤深度值
            dep_data = depth_data.copy()
            dep_data[dep_data > 10000.0] = 0
            dep_data = dep_data * depth_scale
            # 四舍五入保留两位小数
            dep_data = dep_data.round(2)
            
            # 计算dep_data的有效值个数（大于0的值）
            valid_count = (dep_data > 0.0).sum()
            print(f"有效值个数: {valid_count},有效率：{valid_count / dep_data.size :.2f}")
            
            # 获取有效深度值的索引
            valid_indices = np.where(dep_data > 0.0)
            valid_depths = dep_data[valid_indices]
            
            # 创建点云数据
            points_3d = []
            colors = []
            
            for i in range(len(valid_indices[0])):
                u, v = valid_indices[1][i], valid_indices[0][i]  # 像素坐标
                z = valid_depths[i]  # 深度值（米）
                
                # 计算3D坐标
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # 获取RGB颜色值
                color = rgb_image[v, u].tolist()
                
                points_3d.append([x, y, z])
                colors.append(color)
            
            # 转换为numpy数组
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            
            print(f"生成的3D点云数量: {points_3d.shape[0]}")
            if points_3d.shape[0] > 0:
                print(f"3D点云范围 - X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] m")
                print(f"3D点云范围 - Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] m")
                print(f"3D点云范围 - Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] m")
            
            return points_3d, colors
        except Exception as e:
            print(f"从深度图生成点云时出错: {e}")
            return np.array([]), np.array([])
    
    def ply_to_obj(self, ply_file_path, obj_file_path):
        """
        将PLY文件转换为OBJ文件
        
        参数:
            ply_file_path: 输入PLY文件路径
            obj_file_path: 输出OBJ文件路径
            
        返回:
            bool: 是否成功转换
        """
        try:
            # 读取PLY文件
            points, colors = self.read_ply_file(ply_file_path)
            if len(points) == 0:
                print("错误: 无法读取PLY文件或文件为空")
                return False
            
            # 创建OBJ文件
            with open(obj_file_path, 'w', encoding='utf-8') as f:
                # 写入材质库声明
                f.write("mtllib material.mtl\n")
                f.write("usemtl point_material\n")
                
                # 写入顶点信息
                print(f"正在写入{len(points)}个顶点到OBJ文件...")
                for point in points:
                    f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                
                # 写入纹理坐标（如果有颜色信息）
                has_colors = len(colors) > 0 and colors.shape[1] >= 3
                if has_colors:
                    for color in colors:
                        # 将RGB转换为0-1范围的浮点数
                        r = max(0, min(255, int(color[0]))) / 255.0
                        g = max(0, min(255, int(color[1]))) / 255.0
                        b = max(0, min(255, int(color[2]))) / 255.0
                        # OBJ格式中使用顶点颜色作为纹理坐标的替代方案
                        # 注意：标准OBJ不直接支持顶点颜色，这里保存为vt行供参考
                        f.write(f"vt {r:.6f} {g:.6f} {b:.6f}\n")
                
                # 写入顶点法线（简单地使用向上方向作为法线）
                # 这是一个简化处理，实际应用中可能需要计算真实法线
                for _ in range(len(points)):
                    f.write("vn 0.000000 0.000000 1.000000\n")
                
                # 写入点元素
                # OBJ格式本身不直接支持点云，但可以使用顶点列表表示
                # 注意：某些OBJ查看器可能需要特殊处理才能正确显示点云
                # 这里我们将每个点作为单独的顶点列出
                f.write("\n# Point cloud data represented as vertices\n")
                
                # 可选：创建材质文件
                material_file = os.path.join(os.path.dirname(obj_file_path), "material.mtl")
                with open(material_file, 'w', encoding='utf-8') as mtl:
                    mtl.write("newmtl point_material\n")
                    mtl.write("Ka 1.000000 1.000000 1.000000\n")  # 环境光
                    mtl.write("Kd 1.000000 1.000000 1.000000\n")  # 漫反射
                    mtl.write("Ks 0.000000 0.000000 0.000000\n")  # 镜面反射
                    mtl.write("d 1.000000\n")  # 透明度
                    mtl.write("Ns 0.000000\n")  # 高光指数
                    mtl.write("illum 1\n")  # 光照模型
            
            print(f"成功将PLY文件转换为OBJ文件: {obj_file_path}")
            print(f"转换的顶点数量: {len(points)}")
            print(f"材质文件已创建: {os.path.basename(material_file)}")
            return True
        except Exception as e:
            print(f"将PLY转换为OBJ时出错: {e}")
            return False
    
    def _generate_sphere_points(self, center, radius, num_points, color):
        """
        使用黄金螺旋算法生成均匀分布的球体点
        
        参数:
            center: 球体中心点坐标
            radius: 球体半径
            num_points: 生成的点数量
            color: 点的颜色
            
        返回:
            tuple: (points, colors) 生成的球体点和颜色列表
        """
        sphere_points = []
        sphere_colors = []
        
        # 使用黄金螺旋算法生成均匀分布的球体点
        phi = np.pi * (3 - np.sqrt(5))  # 黄金角
        
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y从1到-1
            radius_at_y = np.sqrt(1 - y * y)  # 半径在该y水平上的截面
            
            theta = phi * i  # 黄金角增量旋转
            
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            
            # 缩放并平移到指定中心
            sphere_points.append([
                center[0] + x * radius,
                center[1] + y * radius,
                center[2] + z * radius
            ])
            sphere_colors.append(color)
        
        return sphere_points, sphere_colors
    
    def _generate_arrow_points(self, start_point, end_point, radius, num_points, color):
        """
        生成从start_point指向下一个点方向的短实体箭头
        箭头由实心圆柱体（箭身）和实心圆锥体（箭头）组成
        箭头不连接到下一个点，只表示方向
        
        参数:
            start_point: 箭头起点坐标
            end_point: 箭头指向的终点坐标
            radius: 基础半径参数
            num_points: 生成的点数
            color: 箭头颜色
            
        返回:
            tuple: (points, colors) 生成的箭头点和颜色列表
        """
        points = []
        colors = []
        
        # 计算方向向量
        direction = np.array(end_point) - np.array(start_point)
        length = np.linalg.norm(direction)
        if length == 0:
            # 如果起点和终点重合，返回球体
            return self._generate_sphere_points(start_point, radius, num_points, color)
        
        direction_normalized = direction / length
        
        # 计算正交基（用于生成圆柱体表面）
        if abs(direction_normalized[2]) < 0.9:  # 如果方向不是接近z轴
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        right = np.cross(direction_normalized, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction_normalized)
        up = up / np.linalg.norm(up)
        
        # 设置固定的箭头总长度，不依赖于点之间的距离
        fixed_arrow_length = radius * 3.0  # 箭头总长度为半径的3倍
        
        # 生成实心箭身（圆柱体）
        shaft_length = fixed_arrow_length * 0.5  # 箭身占总长度的50%
        shaft_radius = radius * 0.3  # 箭身半径
        
        # 增加箭身的点数和层数，使其看起来更实心
        num_circles = 15  # 层数
        points_per_radius = 3  # 径向填充层数
        
        for i in range(num_circles + 1):
            t = i / num_circles
            # 只从起点向前延伸，不连接到下一个点
            circle_center = np.array(start_point) + direction_normalized * shaft_length * t
            
            # 径向填充：从中心到外边缘生成多个同心圆
            for r_factor in range(points_per_radius):
                current_radius = shaft_radius * (r_factor + 1) / points_per_radius
                # 外层圆的点数更多，内层圆的点数较少
                current_points_per_circle = max(4, int(6 * (r_factor + 1)))
                
                for j in range(current_points_per_circle):
                    angle = 2 * np.pi * j / current_points_per_circle
                    x = circle_center[0] + current_radius * (math.cos(angle) * right[0] + math.sin(angle) * up[0])
                    y = circle_center[1] + current_radius * (math.cos(angle) * right[1] + math.sin(angle) * up[1])
                    z = circle_center[2] + current_radius * (math.cos(angle) * right[2] + math.sin(angle) * up[2])
                    points.append([x, y, z])
                    colors.append(color)
        
        # 生成实心箭头（圆锥体）
        arrow_length = fixed_arrow_length * 0.5  # 箭头占总长度的50%
        arrow_radius = radius * 0.8  # 箭头半径
        arrow_base = np.array(start_point) + direction_normalized * shaft_length
        
        # 增加箭头的点数和层数，使其看起来更实心
        num_arrow_layers = 10  # 层数
        arrow_points_per_radius = 3  # 径向填充层数
        
        for i in range(num_arrow_layers + 1):
            t = i / num_arrow_layers
            current_radius = arrow_radius * (1 - t)
            # 只从箭身末端向前延伸，不连接到下一个点
            layer_center = arrow_base + direction_normalized * arrow_length * t
            
            # 径向填充：从中心到外边缘生成多个同心椭圆
            for r_factor in range(arrow_points_per_radius):
                current_point_radius = current_radius * (r_factor + 1) / arrow_points_per_radius
                # 外层圆的点数更多，内层圆的点数较少
                current_points_per_layer = max(3, int(5 * (r_factor + 1)))
                
                for j in range(current_points_per_layer):
                    angle = 2 * np.pi * j / current_points_per_layer
                    x = layer_center[0] + current_point_radius * (math.cos(angle) * right[0] + math.sin(angle) * up[0])
                    y = layer_center[1] + current_point_radius * (math.cos(angle) * right[1] + math.sin(angle) * up[1])
                    z = layer_center[2] + current_point_radius * (math.cos(angle) * right[2] + math.sin(angle) * up[2])
                    points.append([x, y, z])
                    colors.append(color)
        
        # 确保箭头顶点被添加
        arrow_tip = np.array(start_point) + direction_normalized * fixed_arrow_length
        points.append(arrow_tip)
        colors.append(color)
        
        # 移除采样机制，保留所有生成的点
        # 不再限制点数，确保箭头呈现完整的实体效果
        
        return points, colors
    
    def convert_labeled_points(self, labeled_points):
        """
        将包含标签的点列表转换为仅包含点坐标的列表，并进行坐标变换
        
        输入格式：[{"point": [x, y, z], "label": "0"}, ...]
        输出格式：[[X, Z, Y], ...]，其中X和Z取反
        
        参数:
            labeled_points: 包含标签的点列表
        
        返回:
            list: 转换后的点坐标列表
        """
        try:
            result = []
            for item in labeled_points:
                if "point" in item and len(item["point"]) >= 3:
                    # 提取点坐标 [x, y, z]
                    x, y, z = item["point"]
                    # 应用变换：[X, Y, Z] -> [-X, -Z, Y]
                    transformed_point = [-x, -z, y]
                    result.append(transformed_point)
                else:
                    print(f"警告: 无效的点数据格式: {item}")
            
            print(f"成功转换 {len(result)} 个带标签的点")
            return result
        except Exception as e:
            print(f"转换带标签的点时出错: {e}")
            return []
    
    def get_next_filename(self, base_path):
        """
        生成递增序号的文件名
        
        参数:
            base_path: 基础文件路径，如 "path/to/file.ply"
        
        返回:
            str: 递增序号的文件路径，如 "path/to/file_1.ply"
        """
        # 分离文件路径、文件名和扩展名
        directory, filename = os.path.split(base_path)
        name_without_ext, ext = os.path.splitext(filename)
        
        # 检查是否已经有带序号的文件
        counter = 1
        new_filename = f"{name_without_ext}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
        
        # 找到下一个可用的序号
        while os.path.exists(new_path):
            counter += 1
            new_filename = f"{name_without_ext}_{counter}{ext}"
            new_path = os.path.join(directory, new_filename)
        
        return new_path


if __name__ == "__main__":
    point_cloud_utils = PointCloudUtils()
    dep_data=[]
    rgb_image=[]
    point_cloud = point_cloud_utils.depth_to_point_cloud(dep_data, rgb_image, 383.19929174573906, 384.76715878730715, 317.944484051631, 231.71115593384292)

    result = point_cloud_utils.save_point_cloud_to_ply(point_cloud[0], point_cloud[1], "./log/20251025_221217/point_cloud.ply")
    print(result)
    annotation_points = [[1,0,0], [1,1,0], [1,2,1], [0,1,1]]
    result = point_cloud_utils.add_annotation_points("./log/20251025_221217/point_cloud.ply", "./log/20251025_221217/point_cloud_mask.ply", annotation_points,radius=0.05)
    print(result)
    result = point_cloud_utils.ply_to_obj("./log/20251025_221217/point_cloud_mask.ply", "./log/20251025_221217/point_cloud_mask.obj")
    print(result)
    
    # 测试convert_labeled_points函数
    labeled_points = [
        {'point': [0.0, 0.0, 0.0], 'label': '0'},
        {'point': [-0.5, 2.0, -0.2], 'label': '1'},
        {'point': [-1.0, 4.0, -0.4], 'label': '2'},
        {'point': [-1.5, 6.0, -0.3], 'label': '3'},
        {'point': [-1.771, 8.475, -0.094], 'label': '4'}
    ]
    converted_points = point_cloud_utils.convert_labeled_points(labeled_points)
    print("转换后的点列表:", converted_points)