import cv2
import numpy as np
import json
import os
from lib.img_utils import ImageUtils
from vls import *

if __name__ == '__main__':
    # VLM和LLM初始化
    vlm, llm = initModel()
    # 设定控制命令
    ctrl_cmd = getCtrlCmd()
    # 获取RGBD数据
    rgb_image, depth_data = getCameraData()
    
    # 使用ImageUtils保存RGB图像
    ImageUtils.save_image(rgb_image, os.path.join('debug', 'rgb_image_preview.jpg'))

    # 获得检测提示词和规划提示词
    detect_prompt, plan_prompt = getPrompt(ctrl_cmd)

    # 第一阶段：vlm根据自然语言控制指令进行目标检测
    detect_result = getDetectBBox(vlm, rgb_image, detect_prompt)
    
    print("检测结果:", detect_result)
    # 使用ImageUtils绘制检测结果并保存
    image_with_boxes = ImageUtils.draw_bounding_boxes(rgb_image, detect_result)
    ImageUtils.save_image(image_with_boxes, os.path.join('debug', 'detection_result_with_boxes.jpg'))
    
    # 第二阶段：lsam服务器进行目标分割，进行目标3D建模
    target3d = get3DTargetModel(rgb_image, detect_result, depth_data)
    print(target3d)
    exit(0)
    # 第三阶段：llm根据目标3d模型进行规划
    waypoints = getPlan(llm, target3d, plan_prompt)
    
    # 输出规划结果
    print("\n规划的航点:")
    for i, waypoint in enumerate(waypoints):
        print(f"航点 {i+1}: {waypoint}")

