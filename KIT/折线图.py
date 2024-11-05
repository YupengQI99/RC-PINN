import matplotlib.pyplot as plt

# 数据
num_data = [64, 94, 124, 154]
mae_case1 = [1.1413, 0.9172, 0.8479, 0.7772]
mre_case1 = [0.3512, 0.2843, 0.2629, 0.2405]
r2_case1 = [0.9829, 0.9904, 0.9912, 0.9923]

mae_case2 = [1.2107, 1.0178, 0.8987, 0.7109]
mre_case2 = [0.3543, 0.2972, 0.2637, 0.2074]
r2_case2 = [0.9904, 0.9913, 0.9940, 0.9955]

mae_case3 = [1.3451, 1.1254, 1.0353, 0.8116]
mre_case3 = [0.3934, 0.3293, 0.300, 0.2292]
r2_case3 = [0.9940, 0.9953, 0.9970, 0.9981]

# 绘制 MAE 折线图
plt.figure(figsize=(10, 6))
plt.plot(num_data, mae_case1, marker='o', label='Case 1')
plt.plot(num_data, mae_case2, marker='o', label='Case 2')
plt.plot(num_data, mae_case3, marker='o', label='Case 3')
plt.xlabel('Number of Data Points')
plt.ylabel('MAE')
plt.title('MAE for Three Cases')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 MRE 折线图
plt.figure(figsize=(10, 6))
plt.plot(num_data, mre_case1, marker='o', label='Case 1')
plt.plot(num_data, mre_case2, marker='o', label='Case 2')
plt.plot(num_data, mre_case3, marker='o', label='Case 3')
plt.xlabel('Number of Data Points')
plt.ylabel('MRE')
plt.title('MRE for Three Cases')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 R2 折线图
plt.figure(figsize=(10, 6))
plt.plot(num_data, r2_case1, marker='o', label='Case 1')
plt.plot(num_data, r2_case2, marker='o', label='Case 2')
plt.plot(num_data, r2_case3, marker='o', label='Case 3')
plt.xlabel('Number of Data Points')
plt.ylabel('R2')
plt.title('R2 for Three Cases')
plt.legend()
plt.grid(True)
plt.show()
