import matplotlib.pyplot as plt
import torch

# 热源位置和大小
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75],[-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])

# 创建图形和轴
fig, ax = plt.subplots(figsize=(2.5, 2.5))

# 设置边界大小为2.5cm x 2.5cm (转换为英寸)
border_size = 2.5 / 2.54  # 将厘米转换为英寸
ax.set_xlim(-border_size, border_size)
ax.set_ylim(-border_size, border_size)
ax.add_patch(plt.Rectangle((-border_size, -border_size), 2*border_size, 2*border_size,
                           fill=None, edgecolor='red', lw=2))  # 红色边框

# 绘制热源
for i, (pos, unit) in enumerate(zip(positions, units)):
    ax.add_patch(plt.Rectangle((pos[0] - unit[0] / 2, pos[1] - unit[1] / 2), unit[0], unit[1],
                               color='salmon', ec='black', lw=1))

# 添加标度尺（以毫米为单位）
ax.annotate('', xy=(0, -border_size), xytext=(0.1, -border_size),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.text(0.05, -border_size - 0.1, '1 mm', ha='center', va='top')

# 自定义绘图（移除轴线，添加标题等）
ax.axis('off')

# 保存为SVG
plt.savefig("final_satellite_motherboard_with_scale_mm.svg", format="svg")

# 显示绘图
plt.show()
