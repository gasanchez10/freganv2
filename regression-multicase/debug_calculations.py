import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load student performance data
train_x = pd.read_csv("../student_performance/data/train_x_1.0.0_continuous.csv")
train_y = pd.read_csv("../student_performance/data/train_y_1.0.0_continuous.csv")
train_t = pd.read_csv("../student_performance/data/train_t_1.0.0_continuous.csv")
test_x = pd.read_csv("../student_performance/data/test_x_1.0.0_continuous.csv")
test_y = pd.read_csv("../student_performance/data/test_y_1.0.0_continuous.csv")

# Load CATE counterfactuals
train_potential = pd.read_csv("./student_performance_cate_results/train_potential_y_1.0.0_continuous_cate.csv")

# Extract counterfactuals
train_cf_cate = []
for i in range(len(train_potential)):
    if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
        train_cf_cate.append(train_potential.iloc[i, 1])
    else:  # Treatment group, counterfactual is Y0
        train_cf_cate.append(train_potential.iloc[i, 0])

train_cf_cate = pd.DataFrame(train_cf_cate, columns=['0'])

print("=== DATA VERIFICATION ===")
print(f"Train factual mean: {train_y.iloc[:, 0].mean():.4f}")
print(f"Train counterfactual mean: {train_cf_cate.iloc[:, 0].mean():.4f}")
print(f"Test mean: {test_y.iloc[:, 0].mean():.4f}")

# Test pure factual (α=1.0)
print("\n=== ALPHA = 1.0 (Pure Factual) ===")
rf_factual = RandomForestRegressor(n_estimators=100, random_state=42)
rf_factual.fit(train_x, train_y.iloc[:, 0])
pred_factual = rf_factual.predict(test_x)
mse_factual = mean_squared_error(test_y, pred_factual)
print(f"MSE with pure factual: {mse_factual:.4f}")

# Test pure counterfactual (α=0.0)
print("\n=== ALPHA = 0.0 (Pure Counterfactual) ===")
rf_cf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cf.fit(train_x, train_cf_cate.iloc[:, 0])
pred_cf = rf_cf.predict(test_x)
mse_cf = mean_squared_error(test_y, pred_cf)
print(f"MSE with pure counterfactual: {mse_cf:.4f}")

print(f"\n=== COMPARISON ===")
print(f"Factual MSE: {mse_factual:.4f}")
print(f"Counterfactual MSE: {mse_cf:.4f}")
print(f"Difference: {mse_factual - mse_cf:.4f}")

if mse_cf < mse_factual:
    print("❌ PROBLEM: Counterfactual predicts better than factual!")
else:
    print("✅ CORRECT: Factual predicts better than counterfactual")