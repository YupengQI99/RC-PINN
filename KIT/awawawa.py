import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import os

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
val_data = pd.read_csv(r'D:\桌面\新选点\绝热验证集.csv')  # 验证集数据文件

def preprocess_data(data):
    x = torch.from_numpy(data['x'].values.astype(np.float32)).reshape(-1, 1)
    y = torch.from_numpy(data['y'].values.astype(np.float32)).reshape(-1, 1)
    T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).reshape(-1, 1)
    # 不进行归一化，直接返回原始数据
    return x, y, T_actual

# 数据预处理
val_x, val_y, val_T_actual = preprocess_data(val_data)

# 创建数据集和数据加载器
val_dataset = TensorDataset(val_x, val_y, val_T_actual)
batch_size = 64  # 可根据需要调整
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 自定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义模型类，与训练时保持一致
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

# 创建模型实例
model = MLP().to(device)

# 加载最佳模型权重
save_dir = 'D:\\Desktop'  # 确保与训练代码中的路径一致
model_save_path = os.path.join(save_dir, 'best_model_weights.pth')

# 检查模型文件是否存在
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f'Model weights loaded from {model_save_path}')
else:
    print(f'Model file not found at {model_save_path}')
    exit()

# 定义评估指标计算函数
def calculate_mae(pred, actual):
    return torch.mean(torch.abs(pred - actual))

def calculate_mre(pred, actual):
    return torch.mean(torch.abs(pred - actual) / torch.abs(actual))

def calculate_r2(pred, actual):
    total_var = torch.sum((actual - torch.mean(actual)) ** 2)
    unexplained_var = torch.sum((actual - pred) ** 2)
    r2_value = 1 - (unexplained_var / total_var)
    return r2_value

# 在验证集上评估模型
model.eval()
all_outputs = []
all_actuals = []
with torch.no_grad():
    for val_batch_x, val_batch_y, val_batch_T_actual in val_loader:
        val_batch_x = val_batch_x.to(device)
        val_batch_y = val_batch_y.to(device)
        val_batch_T_actual = val_batch_T_actual.to(device)

        val_inputs = torch.cat([val_batch_x, val_batch_y], dim=1)
        val_outputs = model(val_inputs)

        all_outputs.append(val_outputs)
        all_actuals.append(val_batch_T_actual)

# 汇总所有预测值和真实值
all_outputs = torch.cat(all_outputs, dim=0)
all_actuals = torch.cat(all_actuals, dim=0)

# 计算评估指标
val_mae = calculate_mae(all_outputs, all_actuals).item()
val_mre = calculate_mre(all_outputs, all_actuals).item()
val_r2 = calculate_r2(all_outputs, all_actuals).item()

print(f'Validation Results:')
print(f'    MAE: {val_mae:.6f}')
print(f'    MRE: {val_mre:.6f}')
print(f'    R² : {val_r2:.6f}')

# 绘制真实值与预测值的对比图
plt.figure(figsize=(10, 7))
plt.scatter(all_actuals.cpu().numpy(), all_outputs.cpu().numpy(), alpha=0.5)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.grid(True)
plt.show()

# 绘制误差分布图
errors = (all_outputs - all_actuals).cpu().numpy()
plt.figure(figsize=(10, 7))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)
plt.show()
