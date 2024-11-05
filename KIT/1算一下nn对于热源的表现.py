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

# 创建模型实例并加载权重
PINN = MLP().to(device)
model_path = r'D:\\桌面\\case1权重\\nn64.pth'

PINN.load_state_dict(torch.load(model_path))
optimizer = torch.optim.Adam(PINN.parameters(), lr=0.001)


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

# 定义热源区域判断函数
def is_chip(x, y, x_locs, y_locs):
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for x_loc, y_loc in zip(x_locs, y_locs):
        is_chip |= (x > x_loc[0]) & (x < x_loc[1]) & (y > y_loc[0]) & (y < y_loc[1])
    return is_chip

# 定义热源区域的 MAE 计算函数
def calculate_chip_mae(pred, actual, is_chip):
    return torch.mean(torch.abs(pred[is_chip] - actual[is_chip]))

# 定义热源区域的 MRE 计算函数
def calculate_chip_mre(pred, actual, is_chip):
    return torch.mean(torch.abs(pred[is_chip] - actual[is_chip]) / torch.abs(actual[is_chip]))

# 热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75], [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

# 验证部分
PINN.eval()
with torch.no_grad():
    val_losses = {
        'val_loss_T': [],
        'val_MAE': [],
        'val_MRE': [],
        'val_R2': [],
        'chip_MAE': [],
        'chip_MRE': []
    }

    T_pred = PINN(torch.cat([test_x, test_y], dim=1))
    T_loss = loss(T_pred, test_T_actual)
    mae_loss_value = calculate_mae(T_pred, test_T_actual)
    mre_loss_value = calculate_mre(T_pred, test_T_actual)
    r2_loss_value = calculate_r2(T_pred, test_T_actual)

    # 判断热源区域
    is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)

    # 计算热源区域的 MAE 和 MRE
    chip_mae_loss_value = calculate_chip_mae(T_pred, test_T_actual, is_chip_data)
    chip_mre_loss_value = calculate_chip_mre(T_pred, test_T_actual, is_chip_data)

    # 记录验证损失
    val_losses['val_loss_T'].append(T_loss.item())
    val_losses['val_MAE'].append(mae_loss_value.item())
    val_losses['val_MRE'].append(mre_loss_value.item())
    val_losses['val_R2'].append(r2_loss_value.item())
    val_losses['chip_MAE'].append(chip_mae_loss_value.item())
    val_losses['chip_MRE'].append(chip_mre_loss_value.item())

    print(f"Validation Losses:")
    for loss_name, loss_values in val_losses.items():
        print(f"    Avg {loss_name}: {loss_values[-1]:.8f}")

# 绘制损失曲线
plt.figure(figsize=(10, 7))
plt.plot(val_losses['val_loss_T'], label='Validation loss_T')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(val_losses['val_MAE'], label='Validation MAE')
plt.plot(val_losses['val_MRE'], label='Validation MRE')
plt.plot(val_losses['val_R2'], label='Validation R2')
plt.plot(val_losses['chip_MAE'], label='Chip Area MAE')
plt.plot(val_losses['chip_MRE'], label='Chip Area MRE')
plt.xlabel('Epochs')
plt.ylabel('MAE/MRE/R2')
plt.title('Validation MAE/MRE/R2 Curves')
plt.legend()
plt.grid(True)
plt.show()
