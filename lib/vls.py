# vls.py - Vision Language System 封装文件
# 封装VLM和LLM模型的初始化和使用函数

import cv2
import json
import numpy as np
from lib.mlm import VLM, LLM
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
    llm = LLM('gpt-5.1',base_url='https://api.zhizengzeng.com/v1')
    # llm = LLM()
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
        latest_data = reader.read_log_data('20251103_104756')
        # 获取RGB图像
        rgb_image = latest_data['color_image']
        # 获取深度数据
        dep_data = latest_data['depth_data']

        return rgb_image, dep_data


def getPrompt(instruction, task="detect", objects_json=""):
    """
    根据控制命令生成检测提示词和规划提示词
    
    Args:
        instruction: 控制命令字符串
        task: 任务类型，默认"detect"
        objects_json: 场景中对象的JSON字符串
    
    Returns:
        tuple: (detect_prompt, plan_prompt) - 检测提示词和规划提示词
    """
    detect_prompt = f"""
    Analyze the drone control instruction: '{instruction}'
    
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

    plan_prompt = f"""
You are a professional UAV 3D path planning system.
Your task is to convert a natural-language flight instruction and a list of 3D objects
into a smooth, safe, executable 3D flight path represented by geometric guidepoints.

Attention: 
- The start point of the path must be the current position of the drone (0.0, 0.0, 0.0).
- Gravity aligns with the y-axis, which is vital for aerial UAV flight.
- Only interpret the INSTRUCTION to the extent necessary to generate a safe path that fulfills the INSTRUCTION. Do not over-interpret or misinterpret the INSTRUCTION. 
- In the UAV's coordinate system, the front of an object has a smaller z-value, while the rear has a larger z-value.
- The ground is y=2.
====================
USER INSTRUCTION
====================
{instruction}

====================
SCENE OBJECTS (JSON LIST)
====================
{objects_json}

Each object in the list represents a solid 3D obstacle.
Treat every object as having volume, approximated by a sphere:
- center: [cx, cy, cz] in meters (right-handed coordinate system)
  * z positive = forward
  * y positive = downward (gravity direction)
- safe_distance: R, the minimum safe distance from the object center to the drone (meters)

The drone and its path MUST NEVER enter inside this safety sphere.
For EVERY guidepoint [x, y, z] and for EVERY object with center [cx, cy, cz] and safe_distance R:
- Let d = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
- You MUST enforce d >= R (meters).
This constraint applies to ALL guidepoints, including the final one.

In addition, consider the polyline path formed by connecting the guidepoints in order.
For EVERY pair of consecutive guidepoints P_i and P_(i+1), and for EVERY object:
- The straight line segment between P_i and P_(i+1) MUST also stay outside the safety sphere.
- That is, the minimum distance between the segment P_i–P_(i+1) and the object center
  MUST be >= R (meters).
The continuous path obtained by linearly interpolating between guidepoints must NEVER cross
or enter the safety sphere of any object.

Do NOT ever place a guidepoint at the object center.
Do NOT ever place a guidepoint with distance d < R to any object center.
Do NOT create line segments that cut through or touch the safety sphere.

====================
SEMANTICS OF NATURAL LANGUAGE INSTRUCTIONS
====================
Interpret typical instructions as follows (always respecting the safety distance rule):

- "fly to the box", "fly to the tree", etc.:
  The drone should move to a point CLOSE TO the target object,
  but still OUTSIDE its safety sphere (d >= R).
  It should NOT go through the object and NOT reach its center.
  For the final guidepoint [xf, yf, zf] and target center [cx, cy, cz]:
  - d = sqrt((xf - cx)^2 + (yf - cy)^2 + (zf - cz)^2) should be as close as reasonably possible
    to R while still satisfying d >= R (do not stay unnecessarily far away).
  - The final point should be located NEAR the object rather than far ABOVE or far BEYOND it:
    * avoid |yf - cy| being much larger than R/2 (do not “fly far over the top”);
    * avoid zf being significantly greater than cz + R (do not “fly far past the object”).
  Intuitively: end up next to the object at approximately the safety distance,
  like “stopping near the object”, not flying over it or far beyond it.

- "fly around the box", "circle around the tree", etc.:
  Generate a circular or arc-like path AROUND the object,
  at a distance >= R from the object center.

- "fly over the box":
  Increase altitude (negative y direction) and pass ABOVE the object,
  always with d >= R to its center.

- "fly past the box":
  Go towards the object and then continue beyond it,
  but still never entering the safety sphere.

Every object is solid and has volume. The path must always go AROUND objects, not THROUGH them.
Do not treat "fly to X" as "fly over X" or "fly past X" unless explicitly stated in the instruction.

====================
PATH OPTIMIZATION / ROUTE SELECTION
====================
Among all paths that satisfy the safety constraints, you should choose a path that:
- Best matches the intent of the control instruction (e.g. "to", "around", "over", "past").
- Is reasonably short in total length (avoid unnecessary detours).
- Is smooth, with gentle changes of direction (no abrupt turns).
- Uses an appropriate number of guidepoints (3–12) without redundancy.

In other words, select a route that is BOTH safe and approximately optimal:
- consistent with the instruction semantics,
- simple and efficient to execute,
- and well-suited for a real UAV controller.

====================
OUTPUT REQUIREMENTS
====================
You must output ONLY a JSON dictionary with EXACTLY the following structure:

{{
  "guidepoints": [
    [x1, y1, z1],
    [x2, y2, z2],
    [x3, y3, z3]
  ],
  "bdtp": "A short explanation (1-3 sentences) of the reasoning behind the chosen path"
}}

STRICT RULES FOR guidepoints:
- guidepoints MUST be a list of 3–12 items.
- Each item MUST be a 3-element list of floats: [x, y, z].
- The FIRST guidepoint MUST be [0.0, 0.0, 0.0], the current drone position.
- The points must form a smooth, simple, realistic path.
- For EVERY guidepoint and EVERY object:
  d = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2) MUST satisfy d >= safe_distance (R).
- For EVERY pair of consecutive guidepoints and EVERY object:
  the straight segment between them MUST also remain at distance >= safe_distance (R)
  from the object center (the continuous path must not cross the safety sphere).
- The path must reflect the user instruction while respecting the safety distance.
- The path should be approximately optimal: safe, short, smooth, and consistent with the instruction.
- Do NOT include velocities, orientations, or comments.
- Do NOT use any object center as a guidepoint.

STRICT RULES FOR bdtp:
- bdtp must be a SHORT textual explanation (1–3 sentences).
- Explain the key planning reasoning:
  user intent, obstacle avoidance (using distance >= safe_distance for both points and segments),
  smoothness, and why this route is a reasonable/optimal choice.
- Do NOT list all coordinates again inside bdtp.
- Do NOT show detailed numeric calculations; just describe the idea qualitatively.
- The bdtp MUST be written in Chinese.

====================
SELF-CHECK BEFORE OUTPUT
====================
Before you output the final JSON, you MUST perform an internal self-check:

1) Check the structure:
   - Ensure the output is a valid JSON object with EXACTLY two keys: "guidepoints" and "bdtp".
   - Ensure "guidepoints" is a list with 3–12 items.
   - Ensure the FIRST guidepoint is exactly [0.0, 0.0, 0.0].
   - Ensure every guidepoint is a list of exactly 3 float numbers.

2) Check safety constraints:
   - For EVERY guidepoint and EVERY object, internally verify d >= safe_distance (R).
   - For EVERY pair of consecutive guidepoints and EVERY object, internally verify that
     the minimum distance from the line segment to the object center is >= safe_distance (R).
   - If any violation is found, you MUST adjust the guidepoints (move or insert points)
     and re-check until all constraints are satisfied.

3) Check instruction semantics:
   - Verify that the overall shape of the path matches the instruction type:
     "fly to", "fly around", "fly over", "fly past", etc.
   - In particular, for "fly to X", ensure the final point is near the object at about
     the safe_distance, not far above it or far beyond it.

4) Check consistency:
   - Ensure that "bdtp" correctly and briefly describes the final path and reasoning,
     without contradicting the actual guidepoints.
   - Do NOT mention the self-check process in the output.
   - Do NOT describe errors or corrections; only present the final, corrected result.

You MUST complete all these self-checks INTERNALLY and silently, then output ONLY the final,
correct JSON that already satisfies all the rules.

====================
FINAL REQUIREMENT
====================
Your final answer MUST be valid JSON with NO extra text before or after it.
NO explanations outside the JSON block.
NO markdown.
NO comments.
"""





    if task == "detect":
        return detect_prompt
    elif task == "plan":
        return plan_prompt
    else:
        raise ValueError("Invalid task. Supported tasks are 'detect' and 'plan'.")

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

def get3DTargetModel(rgb_image, detect_result, depth_data, safe_distance=0.3):
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
    return  processor.process_targets(rgb_image, detect_result, depth_data, safe_distance)



def getPlan(llm, plan_prompt):
    """
    使用LLM根据目标3D模型进行规划
    
    Args:
        llm: LLM模型实例
        target3dmodel: 目标3D模型字典
        plan_prompt: 规划提示词
    
    Returns:
        list: 规划结果列表，包含实际像素坐标
    """
    # 调用LLM进行规划
    response_text = llm.generate_text(
        prompt=plan_prompt
    )
    print(response_text)
    return llm.parse_json_response(response_text)["guidepoints"]


