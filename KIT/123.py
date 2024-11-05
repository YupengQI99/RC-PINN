import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt

# 打印 torch 版本
print(f"PyTorch version: {torch.__version__}")

# 数据加载
train_data = pd.read_csv(r'D:\桌面\折线图\复杂边界34.csv')
test_data = pd.read_csv(r'D:\桌面\新选点\复杂边界测试集.csv')  # 替换为测试集文件的实际路径

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
    opt_result = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=10000)  # 设置 maxiter
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

# 使用Sobol序列生成采样点
num_new_points = 100  # 定义生成的新点数量
sobol = Sobol(d=2, scramble=True)
new_points_scaled = sobol.random(num_new_points)

# 将采样点缩放到原始数据范围
x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()
new_points = np.column_stack((
    new_points_scaled[:, 0] * (x_max - x_min) + x_min,
    new_points_scaled[:, 1] * (y_max - y_min) + y_min
))

# 用高斯过程模型预测新点的温度
new_targets_scaled, sigma = gp.predict(new_points_scaled, return_std=True)

# 反归一化新点的预测温度
new_targets = scaler_T.inverse_transform(new_targets_scaled.reshape(-1, 1)).ravel()

# 选择不确定度最低的25个点和不确定度排序在45到50的点
lowest_uncertainty_idx = np.argsort(sigma)[:25]
mid_uncertainty_idx = np.argsort(sigma)[45:50]

selected_low_points = new_points[lowest_uncertainty_idx]
selected_mid_points = new_points[mid_uncertainty_idx]
selected_low_targets = new_targets[lowest_uncertainty_idx]
selected_mid_targets = new_targets[mid_uncertainty_idx]

# 打印选出的点及其温度
print("Selected points and corresponding temperatures:")
for point, temp in zip(np.concatenate([selected_low_points, selected_mid_points]),
                       np.concatenate([selected_low_targets, selected_mid_targets])):
    print(f"Point: {point}, Temperature: {temp:.4f}")

# 保存生成的数据点和温度到CSV文件
selected_data = np.column_stack((np.concatenate([selected_low_points, selected_mid_points]),
                                 np.concatenate([selected_low_targets, selected_mid_targets])))
output_path = r'D:\\桌面\\高斯复杂34.csv'
np.savetxt(output_path, selected_data, delimiter=',', header='x,y,T', comments='')

print(f"Data saved to {output_path}")

# 可视化原始点、生成的点和选择的点
plt.figure(figsize=(10, 10))

# 绘制原始点
plt.scatter(x_train[:, 0], x_train[:, 1], c='blue', label='Original Points', alpha=0.5)

# 绘制生成的点
plt.scatter(new_points[:, 0], new_points[:, 1], c='green', label='Generated Points', alpha=0.5)

# 绘制选择的不确定度最低的点
plt.scatter(selected_low_points[:, 0], selected_low_points[:, 1], c='red', label='Lowest Uncertainty Points', alpha=0.8)

# 绘制选择的不确定度适中的点
plt.scatter(selected_mid_points[:, 0], selected_mid_points[:, 1], c='orange', label='Mid Uncertainty Points', alpha=0.8)

# 设置图例、标题和坐标轴标签
plt.legend()
plt.title("Visualization of Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# 显示图像
plt.show()
