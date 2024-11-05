import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 创建条形图
plt.figure(figsize=(12, 6))

# 绘制条形图，x 轴为点的索引（从0到len(data)-1），y 轴为不确定度
plt.bar(range(len(data)), data['Uncertainty'], color='skyblue')

# 添加标题和轴标签
plt.title('Bar Plot of Points Sorted by Uncertainty')
plt.xlabel('Point Index (sorted by uncertainty)')
plt.ylabel('Uncertainty')

# 显示网格线
plt.grid(True)

# 保存图像为SVG格式
plt.savefig(r'D:\桌面\newtest\uncertainty_sorted_bar_plot.svg', format='svg')

# 显示图像
plt.show()
