import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load original data (regression_multicase approach)
train_x_orig = pd.read_csv("../synthetic_data/data/train_x_1.0.0_continuous.csv")
train_y_orig = pd.read_csv("../synthetic_data/data/train_y_1.0.0_continuous.csv")
test_x_orig = pd.read_csv("../synthetic_data/data/test_x_1.0.0_continuous.csv")
test_y_orig = pd.read_csv("../synthetic_data/data/test_y_1.0.0_continuous.csv")

# Load CATE results data (fair_model approach)
train_x_cate = pd.read_csv("./synthetic_data_cate_results/train_x_1.0.0_continuous_cate.csv")
train_y_cate = pd.read_csv("./synthetic_data_cate_results/train_y_1.0.0_continuous_cate.csv")
test_x_cate = pd.read_csv("./synthetic_data_cate_results/test_x_1.0.0_continuous_cate.csv")
test_y_cate = pd.read_csv("./synthetic_data_cate_results/test_y_1.0.0_continuous_cate.csv")

print("=== DATA COMPARISON ===")
print(f"Original train shape: {train_x_orig.shape}, CATE train shape: {train_x_cate.shape}")
print(f"Original test shape: {test_x_orig.shape}, CATE test shape: {test_x_cate.shape}")

# Train models
rf_orig = RandomForestRegressor(n_estimators=100, random_state=42)
rf_orig.fit(train_x_orig, train_y_orig.values.ravel())

rf_cate = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cate.fit(train_x_cate, train_y_cate.values.ravel())

# Predict
pred_orig = rf_orig.predict(test_x_orig)
pred_cate = rf_cate.predict(test_x_cate)

# Calculate MSE
mse_orig = mean_squared_error(test_y_orig, pred_orig)
mse_cate = mean_squared_error(test_y_cate, pred_cate)

print(f"\n=== MSE COMPARISON ===")
print(f"Original approach MSE: {mse_orig:.4f}")
print(f"CATE approach MSE: {mse_cate:.4f}")
print(f"Difference: {abs(mse_orig - mse_cate):.6f}")

# Check if data is identical
print(f"\n=== DATA IDENTITY CHECK ===")
print(f"Train X identical: {train_x_orig.equals(train_x_cate)}")
print(f"Train Y identical: {train_y_orig.equals(train_y_cate)}")
print(f"Test X identical: {test_x_orig.equals(test_x_cate)}")
print(f"Test Y identical: {test_y_orig.equals(test_y_cate)}")