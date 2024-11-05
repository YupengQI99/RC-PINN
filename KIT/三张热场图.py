import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

# 设置字体
plt.rcParams['font.family'] = 'Arial'  # 确保使用 Arial 字体
plt.rcParams['font.size'] = 18  # 修改字体大小为18号

# 定义每个 case 对应的测试集路径
case_to_path = {
    'case1': 'D:\\桌面\\case1.csv',
    'case2': 'D:\\桌面\\case2.csv',
    'case3': 'D:\\桌面\\case3.csv'
}

# 创建一个包含三个子图的图形
fig, axes = plt.subplots(1, 3, figsize=(30, 8))

for ax, (case, path) in zip(axes, case_to_path.items()):
    # 读取实际温度数据
    actual_data = pd.read_csv(path)

    # 处理重复数据
    actual_data = actual_data.groupby(['X', 'Y'], as_index=False).mean()

    # 生成密集的网格点用于绘制图像
    x = np.linspace(actual_data['X'].min(), actual_data['X'].max(), 1000)
    y = np.linspace(actual_data['Y'].min(), actual_data['Y'].max(), 1000)
    x_grid, y_grid = np.meshgrid(x, y)

    # 使用 griddata 将数据插值到网格上
    T_grid = griddata((actual_data['X'], actual_data['Y']), actual_data['T'], (x_grid, y_grid), method='cubic')

    # 绘制预测温度图
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4)  # 增加pad值以增大间距
    c = ax.contourf(x_grid, y_grid, T_grid, levels=100, cmap='jet')
    cbar = fig.colorbar(c, cax=cax)
    cbar.ax.set_ylabel('')  # 移除颜色条标签
    cbar.set_ticks(cbar.get_ticks()[::2])  # 只显示每隔一个刻度的标签
    ax.set_title(f'{case} Temperature Distribution', fontsize=18)  # 简化标题
    ax.set_xticks([])  # 移除x轴刻度
    ax.set_yticks([])  # 移除y轴刻度
    ax.set_aspect('equal')  # 确保热场图是正方形

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.3)  # 调整wspace值以减少子图之间的间距

# 保存图像
save_path_svg = 'D:\\桌面\\combined_cases.svg'
save_path_png = 'D:\\桌面\\combined_cases.png'
os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)

# 保存最终的大图
plt.savefig(save_path_svg, format='svg')
plt.savefig(save_path_png, format='png', dpi=300)  # 保存为PNG格式，设置dpi为300以确保清晰度
plt.show()
