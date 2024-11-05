import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import torch

# Load test data
test_file = "D:\\桌面\\新选点\\复杂边界测试集.csv"
test_data = pd.read_csv(test_file)
X_test = test_data[['x', 'y']].values
y_test = test_data['T'].values

# Define models
models = {
    'Random Forest': RandomForestRegressor(),
    'Gaussian Process': GaussianProcessRegressor(),
    'Ridge Regression': Ridge()
}

# Example to filter data within chip area
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])

def is_within_chip_area(x, y, positions, units):
    for pos, unit in zip(positions, units):
        if (pos[0] - unit[0] <= x <= pos[0] + unit[0]) and (pos[1] - unit[1] <= y <= pos[1] + unit[1]):
            return True
    return False

mask = np.array([is_within_chip_area(x, y, positions, units) for x, y in X_test])
X_chip = X_test[mask]
y_chip = y_test[mask]

# Train and evaluate on each dataset
train_files = [
    "D:\\桌面\\折线图\\复杂边界34.csv",
    "D:\\桌面\\新选点\\复杂边界64.csv",
    "D:\\桌面\\新选点\\复杂边界94.csv",

]

for train_file in train_files:
    train_data = pd.read_csv(train_file)
    X_train = train_data[['x', 'y']].values
    y_train = train_data['T'].values

    print(f"Results for training data from {train_file}:")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        results[name] = {'MAE': mae, 'MRE': mre}

        # Calculate for chip area
        y_pred_chip = model.predict(X_chip)
        mae_chip = mean_absolute_error(y_chip, y_pred_chip)
        mre_chip = np.mean(np.abs((y_chip - y_pred_chip) / y_chip)) * 100
        results[name]['Chip MAE'] = mae_chip
        results[name]['Chip MRE'] = mre_chip

    # Print results
    for name, metrics in results.items():
        print(f"{name}: MAE = {metrics['MAE']:.4f}, MRE = {metrics['MRE']:.2f}%")
        print(f"Chip Area: MAE = {metrics['Chip MAE']:.4f}, MRE = {metrics['Chip MRE']:.2f}%")
    print("\n")
