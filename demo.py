from PIL import Image
import numpy as np

# 创建一个480行×640列×3通道的数组，初始化为全0（黑色）
image_array = np.zeros((480, 640, 3), dtype=np.uint8)

# 将所有像素设置为红色（RGB格式：红色值为255，绿色和蓝色值为0）
image_array[:, :, 0] = 255  # R通道
image_array[:, :, 1] = 0    # G通道
image_array[:, :, 2] = 0    # B通道

# 将NumPy数组转换为PIL图像
red_image = Image.fromarray(image_array)

# 显示图像
red_image.show()

# 保存图像到文件
red_image.save('red_image.jpg')
print("纯红色图片已保存为red_image.jpg")

#RGB通道顺序