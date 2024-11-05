import torch
import pandas as pd
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

# 数据加载
test_data = pd.read_csv(r'D:\桌面\新选点\一万个点测试集.csv')

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(data):
    x = torch.from_numpy(data['x'].values.astype(np.float32)).to(device).reshape(-1, 1)
    y = torch.from_numpy(data['y'].values.astype(np.float32)).to(device).reshape(-1, 1)
    T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).to(device).reshape(-1, 1)
    return x, y, T_actual

# 数据预处理
test_x, test_y, test_T_actual = preprocess_data(test_data)

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

    def forward(self, x):
        return self.net(x)

# 创建模型并加载权重
PINN = MLP().to(device)
PINN1 = MLP().to(device)
model_path_PINN = r'D:\\桌面\\折线图\\无用途四边等温79测试复杂无方块.pth'
model_path_PINN1 = r'D:\\桌面\\折线图\\1无用途四边等温79测试复杂无方块.pth'
PINN.load_state_dict(torch.load(model_path_PINN))
PINN1.load_state_dict(torch.load(model_path_PINN1))

# 损失函数
loss = nn.MSELoss()

# 计算 MAE
def calculate_mae(pred, actual):
    return torch.mean(torch.abs(pred - actual))

# 计算 MRE
def calculate_mre(pred, actual):
    return torch.mean(torch.abs(pred - actual) / torch.abs(actual))

# 计算 R² 分数
def calculate_r2(pred, actual):
    total_var = torch.sum((actual - torch.mean(actual)) ** 2)
    unexplained_var = torch.sum((actual - pred) ** 2)
    r2_value = 1 - (unexplained_var / total_var)
    return r2_value

# 计算热源区域的 MAE
def calculate_chip_mae(pred, actual, is_chip):
    return torch.mean(torch.abs(pred[is_chip] - actual[is_chip]))

# 定义热源区域的 MRE 计算函数
def calculate_chip_mre(pred, actual, is_chip):
    return torch.mean(torch.abs(pred[is_chip] - actual[is_chip]) / torch.abs(actual[is_chip]))

# 储存热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

# 判定每个点是否在 chip 位置范围内
def is_chip(x, y, x_locs, y_locs, margin=0.01):
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        mask = (x >= (x_min - margin)) & (x <= (x_max + margin)) & (y >= (y_min - margin)) & (y <= (y_max + margin))
        is_chip = is_chip | mask
    return is_chip

# 进行预测并计算性能指标
PINN.eval()
PINN1.eval()
with torch.no_grad():
    data_batch = {
        'x': test_x,
        'y': test_y,
        'T_actual': test_T_actual
    }

    # 判定每个点是否在 chip 位置范围内
    is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)
    T_pred_chip = PINN(torch.cat([test_x, test_y], dim=1))
    T_pred_non_chip = PINN1(torch.cat([test_x, test_y], dim=1))
    T_pred = torch.where(is_chip_data, T_pred_chip, T_pred_non_chip)

    # 计算总体 MAE, MRE, R²
    mae_loss_value = calculate_mae(T_pred, test_T_actual)
    mre_loss_value = calculate_mre(T_pred, test_T_actual)
    r2_loss_value = calculate_r2(T_pred, test_T_actual)

    # 计算热源区域的 MAE
    chip_mae_loss_value = calculate_chip_mae(T_pred, test_T_actual, is_chip_data)
    chip_mre_loss_value = calculate_chip_mre(T_pred, test_T_actual, is_chip_data)

    print(f"Test MAE: {mae_loss_value.item():.8f}")
    print(f"Test MRE: {mre_loss_value.item():.8f}")
    print(f"Test R²: {r2_loss_value.item():.8f}")
    print(f"Chip Area MAE: {chip_mae_loss_value.item():.8f}")
    print(f"Chip Area MAE: {chip_mre_loss_value.item():.8f}")

# 绘制预测结果
plt.figure(figsize=(10, 7))
plt.scatter(test_x.cpu(), test_y.cpu(), c=T_pred.cpu().numpy(), cmap='coolwarm', marker='o', label='Predicted')
plt.scatter(test_x.cpu(), test_y.cpu(), c=test_T_actual.cpu().numpy(), cmap='viridis', marker='x', label='Actual')
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted vs Actual Temperatures')
plt.legend()
plt.grid(True)
plt.show()
