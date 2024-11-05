import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 使用散点图，X 轴和 Y 轴表示点的坐标，颜色表示温度，大小表示不确定度
plt.figure(figsize=(10, 8))

# 绘制散点图，X和Y是点的坐标，颜色表示温度，大小表示不确定度
sc = plt.scatter(data['x'], data['y'], c=data['Temperature'],
                 cmap='coolwarm', s=data['Uncertainty']*500, alpha=0.7)

# 添加颜色条，表示温度
cbar = plt.colorbar(sc)
cbar.set_label('Temperature')

# 添加标题和坐标轴标签
plt.title('Scatter Plot of Selected Points by Temperature and Uncertainty')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 显示网格
plt.grid(True)

# 保存图像为SVG格式
plt.savefig(r'D:\桌面\newtest\coordinate_temperature_uncertainty_scatter.svg', format='svg')

# 显示图像
plt.show()
