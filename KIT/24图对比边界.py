import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义模型类
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# 储存热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

# 判定每个点是否在芯片位置范围内
def is_chip(x, y, x_locs, y_locs, margin=0.001):
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        mask = (x >= (x_min - margin)) & (x <= (x_max + margin)) & (y >= (y_min - margin)) & (y <= (y_max + margin))
        is_chip = is_chip | mask
    return is_chip

# 判定每个点是否在交界处
def is_boundary(x, y, x_locs, y_locs, boundary_width=0.05):
    is_boundary = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        boundary_mask = (
            ((x >= (x_min - boundary_width)) & (x <= (x_min + boundary_width))) |
            ((x >= (x_max - boundary_width)) & (x <= (x_max + boundary_width))) |
            ((y >= (y_min - boundary_width)) & (y <= (y_min + boundary_width))) |
            ((y >= (y_max - boundary_width)) & (y <= (y_max + boundary_width)))
        )
        is_boundary = is_boundary | boundary_mask
    return is_boundary & ~is_chip(x, y, x_locs, y_locs)

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18  # 修改字体大小为18号

# 创建绘图
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 调整子图间距
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 减少子图之间的距离

# 选择一个case和数据量进行详细对比
case = 'case1'
points = 64

# 定义每个 case 对应的测试集路径
case_to_path = {
    'case1': 'D:\\桌面\\新选点\\一万个点测试集.csv',
    'case2': 'D:\\桌面\\新选点\\绝热测试集.csv',
    'case3': 'D:\\桌面\\新选点\\复杂边界测试集.csv'
}

# 读取实际温度数据
actual_data_path = case_to_path[case]
actual_data = pd.read_csv(actual_data_path)

# 处理重复数据
actual_data = actual_data.groupby(['x', 'y'], as_index=False).mean()

# 使用模型预测实际值位置的温度
test_x_actual = torch.tensor(actual_data['x'].values, dtype=torch.float32).reshape(-1, 1).to(device)
test_y_actual = torch.tensor(actual_data['y'].values, dtype=torch.float32).reshape(-1, 1).to(device)
input_actual = torch.cat([test_x_actual, test_y_actual], dim=1)

# 创建模型并加载权重
PINN = MLP().to(device)
PINN1 = MLP().to(device)
NN = MLP().to(device)
model_path_PINN = f'D:\\桌面\\{case}权重\\pinn{points}.pth'
model_path_PINN1 = f'D:\\桌面\\{case}权重\\1pinn{points}.pth'
model_path_NN = f'D:\\桌面\\{case}权重\\nn{points}.pth'

# 加载模型权重
PINN.load_state_dict(torch.load(model_path_PINN))
PINN1.load_state_dict(torch.load(model_path_PINN1))
NN.load_state_dict(torch.load(model_path_NN))

# 使用模型预测实际值位置的温度
with torch.no_grad():
    is_chip_data_actual = is_chip(test_x_actual, test_y_actual, x_loc, y_loc)
    T_pred_chip_actual = PINN(input_actual)
    T_pred_non_chip_actual = PINN1(input_actual)
    T_pred_actual = torch.where(is_chip_data_actual, T_pred_chip_actual, T_pred_non_chip_actual)
    T_pred_nn_actual = NN(input_actual)

# 计算绝对误差
absolute_error_pinn = np.abs(T_pred_actual.cpu().numpy().flatten() - actual_data['T'].values)
absolute_error_nn = np.abs(T_pred_nn_actual.cpu().numpy().flatten() - actual_data['T'].values)

# 判定交界处的点
boundary_mask = is_boundary(test_x_actual, test_y_actual, x_loc, y_loc)

# 绘制交界处的PINN绝对误差图
ax_pinn = axes[0]
divider_pinn = make_axes_locatable(ax_pinn)
cax_pinn = divider_pinn.append_axes("right", size="5%", pad=0.1)
c_pinn = ax_pinn.tricontourf(actual_data['x'][boundary_mask.cpu().numpy()], actual_data['y'][boundary_mask.cpu().numpy()], absolute_error_pinn[boundary_mask.cpu().numpy()], levels=100, cmap='jet')
cbar_pinn = fig.colorbar(c_pinn, cax=cax_pinn)
cbar_pinn.ax.set_ylabel('Absolute Error')
ax_pinn.set_title('PINN Absolute Error at Boundary')
ax_pinn.set_xlim(-1, 1)
ax_pinn.set_ylim(-1, 1)
ax_pinn.set_aspect('equal')

# 绘制交界处的NN绝对误差图
ax_nn = axes[1]
divider_nn = make_axes_locatable(ax_nn)
cax_nn = divider_nn.append_axes("right", size="5%", pad=0.1)
c_nn = ax_nn.tricontourf(actual_data['x'][boundary_mask.cpu().numpy()], actual_data['y'][boundary_mask.cpu().numpy()], absolute_error_nn[boundary_mask.cpu().numpy()], levels=100, cmap='jet')
cbar_nn = fig.colorbar(c_nn, cax=cax_nn)
cbar_nn.ax.set_ylabel('Absolute Error')
ax_nn.set_title('NN Absolute Error at Boundary')
ax_nn.set_xlim(-1, 1)
ax_nn.set_ylim(-1, 1)
ax_nn.set_aspect('equal')

# 绘制交界处的误差差值图 (PINN - NN)
ax_diff = axes[2]
divider_diff = make_axes_locatable(ax_diff)
cax_diff = divider_diff.append_axes("right", size="5%", pad=0.1)
error_diff = absolute_error_pinn - absolute_error_nn
c_diff = ax_diff.tricontourf(actual_data['x'][boundary_mask.cpu().numpy()], actual_data['y'][boundary_mask.cpu().numpy()], error_diff[boundary_mask.cpu().numpy()], levels=100, cmap='bwr')
cbar_diff = fig.colorbar(c_diff, cax=cax_diff)
cbar_diff.ax.set_ylabel('Error Difference (PINN - NN)')
ax_diff.set_title('Error Difference at Boundary')
ax_diff.set_xlim(-1, 1)
ax_diff.set_ylim(-1, 1)
ax_diff.set_aspect('equal')

# 确保保存路径存在
save_path_svg = 'D:\\桌面\\科研绘图_boundary_comparison.svg'
save_path_png = 'D:\\桌面\\科研绘图_boundary_comparison.png'
os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)

# 保存最终的大图
fig.savefig(save_path_svg, format='svg')
fig.savefig(save_path_png, format='png', dpi=300)  # 保存为PNG格式，设置dpi为300以确保清晰度
plt.show()
