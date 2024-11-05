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
train_data = pd.read_csv(r'D:\桌面\新选点\复杂边界64.csv')
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

# 根据不确定度选择点
num_new_points = 200  # 定义生成的新点数量
x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()

# 生成新的数据点, 将新点范围设定为与原始数据一致
new_points = np.column_stack((np.random.uniform(x_min, x_max, num_new_points),
                              np.random.uniform(y_min, y_max, num_new_points)))

# 对新点进行归一化
new_points_scaled = scaler_x.transform(new_points)

# 用高斯过程模型预测新点的温度
new_targets_scaled, sigma = gp.predict(new_points_scaled, return_std=True)

# 反归一化新点的预测温度
new_targets = scaler_T.inverse_transform(new_targets_scaled.reshape(-1, 1)).ravel()

# 将生成的点按不确定度进行排序
sorted_indices = np.argsort(sigma)
new_points_sorted = new_points[sorted_indices]
new_targets_sorted = new_targets[sorted_indices]
sigma_sorted = sigma[sorted_indices]

# 打印按不确定度排序后的200个新数据点及其对应的温度和不确定度
print("Generated 200 new points with corresponding temperatures and uncertainties (sorted by uncertainty):")
for i, (point, temp, uncertainty) in enumerate(zip(new_points_sorted, new_targets_sorted, sigma_sorted)):
    print(f"Point {i+1}: {point}, Temperature: {temp:.4f}, Uncertainty: {uncertainty:.4f}")

# 创建数据表格展示生成的200个点的均值和不确定度
new_data_with_uncertainty = pd.DataFrame({
    'x': new_points_sorted[:, 0],
    'y': new_points_sorted[:, 1],
    'Temperature': new_targets_sorted,
    'Uncertainty': sigma_sorted
})

# 保存生成的点和不确定度到CSV文件
output_path = r'D:\桌面\newtest\混合64个点_排序1.csv'
new_data_with_uncertainty.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")

# 可视化生成的点
plt.figure(figsize=(10, 10))

# 绘制生成的点
plt.scatter(new_points_sorted[:, 0], new_points_sorted[:, 1], c='green', label='Generated Points', alpha=0.5)

# 设置图例、标题和坐标轴标签
plt.legend()
plt.title("Visualization of Generated Points (Sorted by Uncertainty)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# 显示图像
plt.show()
