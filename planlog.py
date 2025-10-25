# 示例: 使用logread库读取和显示日志数据
from lib.logread import create_reader, list_all_logs

# 创建日志读取器
reader = create_reader('./log')

# 列出所有可用的日志
list_all_logs(show_details=True)

# 读取最新的日志数据
latest_data = reader.read_latest_log()

# 显示图像
if latest_data['color_image'] is not None and latest_data['depth_visualization'] is not None:
    print(f"\n彩色图像形状: {latest_data['color_image'].shape}")
    print(f"深度可视化图像形状: {latest_data['depth_visualization'].shape}")
    # reader.display_images(latest_data['color_image'], latest_data['depth_visualization'])

# 分析深度数据
if latest_data['depth_data'] is not None:
    depth_stats = reader.analyze_depth_data(latest_data['depth_data'])
    print("\n深度数据统计信息:")
    print(f"  形状: {depth_stats['shape']}")
    print(f"  有效像素: {depth_stats['valid_pixels']} / {depth_stats['total_pixels']}")
    if depth_stats['has_valid_data']:
        print(f"  最大深度: {depth_stats['max_depth']*0.001:.2f} m")
        print(f"  最小深度: {depth_stats['min_depth']*0.001:.2f} m")
        print(f"  平均深度: {depth_stats['mean_depth']*0.001:.2f} m")

# 读取特定日志
# specific_data = reader.read_log_data('20251025_214925')