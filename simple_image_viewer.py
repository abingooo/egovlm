import os
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

# 图像目录
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')

@app.route('/')
def index():
    # HTML模板，直接显示目标图像
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>图像查看器</title>
        <meta charset="utf-8">
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            .image-container {
                max-width: 100%;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            img {
                max-width: 100%;
                max-height: 80vh;
                border-radius: 4px;
            }
            .controls {
                margin-top: 20px;
            }
            button {
                padding: 10px 20px;
                margin: 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            .zoom-controls {
                margin-top: 15px;
            }
            #zoom-value {
                margin: 0 10px;
            }
        </style>
    </head>
    <body>
        <h1>图像查看器</h1>
        
        <div class="image-container">
            <img id="target-image" src="/images/target2d_visual_in_rgbimage.jpg" alt="目标图像">
        </div>
        
        <div class="controls">
            <button onclick="rotateImage(90)">旋转 90°</button>
            <button onclick="rotateImage(-90)">旋转 -90°</button>
            <button onclick="resetImage()">重置</button>
            
            <div class="zoom-controls">
                <button onclick="zoomImage(-0.1)">缩小</button>
                <span id="zoom-value">100%</span>
                <button onclick="zoomImage(0.1)">放大</button>
            </div>
        </div>
        
        <script>
            let rotation = 0;
            let scale = 1;
            const image = document.getElementById('target-image');
            const zoomValue = document.getElementById('zoom-value');
            
            function updateTransform() {
                image.style.transform = `rotate(${rotation}deg) scale(${scale})`;
                zoomValue.textContent = `${Math.round(scale * 100)}%`;
            }
            
            function rotateImage(degrees) {
                rotation += degrees;
                updateTransform();
            }
            
            function zoomImage(factor) {
                scale = Math.max(0.1, Math.min(5, scale + factor));
                updateTransform();
            }
            
            function resetImage() {
                rotation = 0;
                scale = 1;
                updateTransform();
            }
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(html_content)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

def main():
    print(f"简单图像查看器启动中...")
    print(f"图像目录: {IMAGE_DIR}")
    print(f"访问 http://localhost:5000 查看图像")
    print(f"按 Ctrl+C 停止服务")
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()