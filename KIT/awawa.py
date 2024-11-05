import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
# 请将以下路径替换为您的训练数据和验证数据文件路径
train_data = pd.read_csv(r'D:\桌面\新选点\绝热64.csv')
val_data = pd.read_csv(r'D:\桌面\新选点\绝热测试集.csv')  # 假设验证集数据文件名为 '验证集.csv'

def preprocess_data(data):
    x = torch.from_numpy(data['x'].values.astype(np.float32)).reshape(-1, 1)
    y = torch.from_numpy(data['y'].values.astype(np.float32)).reshape(-1, 1)
    T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).reshape(-1, 1)
    # 不进行归一化，直接返回原始数据
    return x, y, T_actual

# 数据预处理
train_x, train_y, train_T_actual = preprocess_data(train_data)
val_x, val_y, val_T_actual = preprocess_data(val_data)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_x, train_y, train_T_actual)
val_dataset = TensorDataset(val_x, val_y, val_T_actual)

batch_size = 64  # 可根据需要调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# 创建模型实例
model = MLP().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

# 训练循环
num_epochs = 10000  # 可根据需要调整
train_losses = []
val_losses = []
val_mae_list = []
val_mre_list = []
val_r2_list = []

best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y, batch_T_actual in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_T_actual = batch_T_actual.to(device)

        # 将 x 和 y 拼接成输入
        inputs = torch.cat([batch_x, batch_y], dim=1)
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, batch_T_actual)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_actuals = []
    with torch.no_grad():
        for val_batch_x, val_batch_y, val_batch_T_actual in val_loader:
            val_batch_x = val_batch_x.to(device)
            val_batch_y = val_batch_y.to(device)
            val_batch_T_actual = val_batch_T_actual.to(device)

            val_inputs = torch.cat([val_batch_x, val_batch_y], dim=1)
            val_outputs = model(val_inputs)
            val_loss_batch = criterion(val_outputs, val_batch_T_actual)
            val_loss += val_loss_batch.item() * val_batch_x.size(0)

            all_outputs.append(val_outputs)
            all_actuals.append(val_batch_T_actual)

    val_loss /= len(val_dataset)
    val_losses.append(val_loss)

    # 计算评估指标
    all_outputs = torch.cat(all_outputs, dim=0)
    all_actuals = torch.cat(all_actuals, dim=0)

    # 不需要反归一化，直接计算评估指标
    val_mae = calculate_mae(all_outputs, all_actuals).item()
    val_mre = calculate_mre(all_outputs, all_actuals).item()
    val_r2 = calculate_r2(all_outputs, all_actuals).item()

    val_mae_list.append(val_mae)
    val_mre_list.append(val_mre)
    val_r2_list.append(val_r2)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.6f}, '
          f'Val Loss: {val_loss:.6f}, '
          f'Val MAE: {val_mae:.6f}, '
          f'Val MRE: {val_mre:.6f}, '
          f'Val R2: {val_r2:.6f}')

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved at epoch {epoch + 1}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 7))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# 绘制验证集上的 MAE、MRE 和 R² 曲线
plt.figure(figsize=(10, 7))
plt.plot(range(1, num_epochs + 1), val_mae_list, label='Validation MAE')
plt.plot(range(1, num_epochs + 1), val_mre_list, label='Validation MRE')
plt.plot(range(1, num_epochs + 1), val_r2_list, label='Validation R2')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Validation MAE, MRE, and R² Curves')
plt.legend()
plt.grid(True)
plt.show()

# 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
