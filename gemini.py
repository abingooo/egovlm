import os
import base64
import json
import requests

def generate_content(image_path, prompt="Caption this image.", api_key="sk-zk26f90a8ef46c6589207af1a58b11c4e4a68eca448256d6", base_url="https://api.zhizengzeng.com/google/v1beta/models/gemini-2.0-flash:generateContent"):
    """
    调用Gemini API生成图像的描述
    
    Args:
        image_path: 图像文件路径
        prompt: 生成描述的提示文本
        api_key: API密钥（如果不提供，将尝试从环境变量GOOGLE_API_KEY获取）
        base_url: API基础URL
        
    Returns:
        图像的描述文本
    """
    # 获取API密钥，如果未提供则从环境变量获取
    if not api_key:
        api_key = os.environ.get('GOOGLE_API_KEY', '')
    
    # 检查API密钥
    if not api_key:
        raise ValueError("请提供API密钥或设置GOOGLE_API_KEY环境变量")
    
    # 读取图像文件并进行base64编码
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 构建请求URL
    url = f"{base_url}?key={api_key}"
    
    # 构建请求体
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",  # 假设图像是JPEG格式
                        "data": encoded_image
                    }
                },
                {"text": prompt},
            ]
        }]
    }
    
    # 发送请求
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # 检查响应
    if response.status_code == 200:
        result = response.json()
        # 提取生成的文本内容
        try:
            # 根据Gemini API响应格式提取文本
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text
        except (KeyError, IndexError) as e:
            return f"解析响应时出错: {e}, 原始响应: {result}"
    else:
        return f"请求失败: 状态码 {response.status_code}, 错误信息: {response.text}"


print(generate_content("./log/rgb_image.jpg", "请描述这张图片"))
