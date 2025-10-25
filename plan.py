from lib.rosrgbd import ROSRGBD



if __name__ == '__main__':
    rgbd = ROSRGBD(log_level='error')

    # 获取RGBD数据
    color_image, depth_image, status = rgbd.getRGBD()

    # 可选：保存图像
    rgbd.save_images()

    # 完成后关闭
    rgbd.shutdown()