import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

# 打印 torch 版本
print(f"PyTorch version: {torch.__version__}")

# 数据加载
train_data = pd.read_csv(r'D:\桌面\折线图\两边绝热34.csv')
test_data = pd.read_csv(r'D:\桌面\新选点\绝热测试集.csv')

# 数据预处理
def preprocess_data(data):
    x = data['x'].values.reshape(-1, 1)
    y = data['y'].values.reshape(-1, 1)
    T = data['T'].values.reshape(-1, 1)
    return np.hstack((x, y)), T

x_train, T_train_actual = preprocess_data(train_data)
x_test, T_test_actual = preprocess_data(test_data)

# 数据归一化
scaler_x = MinMaxScaler()
scaler_T = MinMaxScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
T_train_actual_scaled = scaler_T.fit_transform(T_train_actual)

x_test_scaled = scaler_x.transform(x_test)

# 增大核函数参数的上限
kernel = C(1.0, (1e-4, 1e6)) * RBF([1.0, 1.0], (1e-4, 1e3))

# 自定义优化器函数，用于设置 max_iter
def custom_optimizer(obj_func, initial_theta, bounds):
    opt_result = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=10000)
    return opt_result[0], opt_result[1]

# 使用自定义优化器
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=custom_optimizer)
gp.fit(x_train_scaled, T_train_actual_scaled.ravel())

# 对测试集进行预测
T_test_pred_scaled, sigma_test = gp.predict(x_test_scaled, return_std=True)

# 反归一化预测结果
T_test_pred = scaler_T.inverse_transform(T_test_pred_scaled.reshape(-1, 1)).ravel()

# 计算评估指标
mre = np.mean(np.abs((T_test_pred - T_test_actual.ravel()) / T_test_actual.ravel()))
mae = mean_absolute_error(T_test_actual, T_test_pred)
r2 = r2_score(T_test_actual, T_test_pred)

print(f"Mean Relative Error (MRE) on the test set: {mre:.4f}")
print(f"Mean Absolute Error (MAE) on the test set: {mae:.4f}")
print(f"R² Score on the test set: {r2:.4f}")

# 根据不确定度选择点
num_new_points = 200
x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()

# 生成新的数据点
new_points = np.column_stack((np.random.uniform(x_min, x_max, num_new_points),
                              np.random.uniform(y_min, y_max, num_new_points)))

# 对新点进行归一化
new_points_scaled = scaler_x.transform(new_points)

# 用高斯过程模型预测新点的温度
new_targets_scaled, sigma = gp.predict(new_points_scaled, return_std=True)

# 反归一化新点的预测温度
new_targets = scaler_T.inverse_transform(new_targets_scaled.reshape(-1, 1)).ravel()

# 按不确定度从低到高排序
sorted_idx = np.argsort(sigma)
sorted_points = new_points[sorted_idx]
sorted_targets = new_targets[sorted_idx]
sorted_sigma = sigma[sorted_idx]

# 保存生成的数据点和温度到CSV文件
sorted_data = np.column_stack((sorted_points, sorted_targets, sorted_sigma))
output_path = r'D:\\桌面\\高斯绝热34.csv'
np.savetxt(output_path, sorted_data, delimiter=',', header='x,y,T,uncertainty', comments='')

print(f"Data saved to {output_path}")

# 可视化原始点、生成的点和选择的点
plt.figure(figsize=(10, 10))

# 绘制原始点
plt.scatter(x_train[:, 0], x_train[:, 1], c='blue', label='Original Points', alpha=0.5)

# 绘制生成的点
plt.scatter(new_points[:, 0], new_points[:, 1], c='green', label='Generated Points', alpha=0.5)

# 绘制选择的点
plt.scatter(sorted_points[:30, 0], sorted_points[:30, 1], c='red', label='Top 30 Points by Uncertainty', alpha=0.8)

# 设置图例、标题和坐标轴标签
plt.legend()
plt.title("Visualization of Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# 显示图像
plt.show()
