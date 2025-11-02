import os
import numpy as np
import math
import cv2 as cv
import traceback
from lib.cloud_utils import PointCloudUtils
from lib.logread import create_reader

# 创建日志读取器
reader = create_reader('./log')

# 读取最新的日志数据
latest_data = reader.read_latest_log()
# 获取RGB图像
rgb_image = latest_data['color_image']
# 获取深度数据
dep_data = latest_data['depth_data']

point_cloud_utils = PointCloudUtils()

# point_cloud = point_cloud_utils.depth_to_point_cloud(dep_data, rgb_image, 383.19929174573906, 384.76715878730715, 317.944484051631, 231.71115593384292)
# result = point_cloud_utils.save_point_cloud_to_ply(point_cloud[0], point_cloud[1], "./log/20251025_221217/point_cloud.ply")

annotation_points = [[0.0, 0.0, 0.0], [0.5, -0.2, 2.0], [1.0, -0.4, 4.0], [1.5, -0.3, 6.0],[1.771, -0.094, 8.475]]
result = point_cloud_utils.add_annotation_points("./log/20251025_221217/point_cloud.ply", "./log/20251025_221217/point_cloud_mask.ply", annotation_points,radius=0.1,points_per_arrow=4000)
print(result)

