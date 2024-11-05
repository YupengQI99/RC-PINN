import matplotlib.pyplot as plt
import numpy as np

# Data sizes and methods
data_sizes = [34, 64, 94]
methods = ['PINN', 'NN', 'Gaussian Regression', 'Random Forest']

# Updated data for three different plots
mae_values_1 = {
    34: [1.1084, 4.1265, 2.7570, 4.6999],
    64: [0.8977, 1.9297, 2.0439, 3.3923],
    94: [0.8440, 1.4642, 1.4552, 2.5711]
}

mae_values_2 = {
    34: [1.1084, 4.2132, 2.6969, 4.2911],
    64: [0.8977, 2.1982, 1.9339, 3.1571],
    94: [0.8440, 1.9759, 1.8918, 2.7589]
}

mae_values_3 = {
    34: [3.0198, 5.9924, 4.9340, 4.4454],
    64: [1.2895, 1.9807, 2.0505, 3.6381],
    94: [1.1489, 1.9204, 2.0303, 3.1918]
}

# Colors specified by the user
colors_updated = ['dodgerblue', 'orange', 'limegreen', 'red']

# Creating a figure with 1 row and 3 columns for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Width of bars and positions for the groups
bar_width = 0.2
index = np.arange(len(data_sizes))

# Plotting each of the three sets of MAE values on separate subplots with black edges for bars
for ax, mae_values, title in zip(axes, [mae_values_1, mae_values_2, mae_values_3],
                                 ['Plot 1', 'Plot 2', 'Plot 3']):
    # Plotting each method with specified colors and black edge color
    for i, method in enumerate(methods):
        mae_scores = [mae_values[size][i] for size in data_sizes]
        ax.bar(index + i * bar_width, mae_scores, width=bar_width, label=method,
               color=colors_updated[i], edgecolor='black')

    # Adding labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Data Size')
    ax.set_ylabel('MAE')
    ax.set_title(title)
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(data_sizes)
    ax.legend()

# Adjusting layout for better visual appearance
plt.tight_layout()

# Saving the plot as an SVG file
fig.savefig("D:\\桌面\\MAE_Histograms.svg", format='svg')

# Displaying the plot
plt.show()
