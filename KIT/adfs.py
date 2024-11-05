from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 创建3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# X, Y 是点的坐标，Z 轴是表示不确定度，颜色表示温度
sc = ax.scatter(data['x'], data['y'], data['Uncertainty'], c=data['Temperature'],
                cmap='coolwarm', s=data['Uncertainty']*500, alpha=0.7)

# 添加颜色条，表示温度
cbar = fig.colorbar(sc)
cbar.set_label('Temperature')

# 添加标题和轴标签
ax.set_title('3D Scatter Plot of Points by Temperature and Uncertainty')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Uncertainty')

# 保存图像为SVG格式
plt.savefig(r'D:\桌面\newtest\3D_temperature_uncertainty_scatter.svg', format='svg')

# 显示图像
plt.show()
