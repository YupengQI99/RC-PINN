import matplotlib.pyplot as plt
import numpy as np

# Data sizes and methods
data_sizes = [34, 64, 94]
methods = ['PINN', 'NN', 'Gaussian Regression', 'Random Forest']

# MAE values for each method at each data size
mae_values_updated = {
    34: [3.0198, 5.9924, 4.9340, 4.4454],
    64: [1.2895, 1.9807, 2.0505, 3.6381],
    94: [1.1489, 1.9204, 2.0303, 3.1918]
}

# Creating figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Width of bars and positions for the groups
bar_width = 0.2
index = np.arange(len(data_sizes))

# Colors specified by the user
colors_updated = ['dodgerblue', 'orange', 'limegreen', 'red']

# Plotting each method with specified colors
for i, method in enumerate(methods):
    mae_scores = [mae_values_updated[size][i] for size in data_sizes]
    ax.bar(index + i * bar_width, mae_scores, width=bar_width, label=method, color=colors_updated[i])

# Adding labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Data Size')
ax.set_ylabel('MAE')
ax.set_title('Updated MAE by Method and Data Size with Specified Colors')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(data_sizes)
ax.legend()

# Saving the plot as an SVG file
fig.savefig("D:\\桌面\\Updated_MAE_Histogram3.svg", format='svg')

# Displaying the plot
plt.show()
