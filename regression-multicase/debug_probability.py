import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
train_y = pd.read_csv("../student_performance/data/train_y_1.0.0_continuous.csv")
train_t = pd.read_csv("../student_performance/data/train_t_1.0.0_continuous.csv")
train_x = pd.read_csv("../student_performance/data/train_x_1.0.0_continuous.csv")
test_x = pd.read_csv("../student_performance/data/test_x_1.0.0_continuous.csv")
test_y = pd.read_csv("../student_performance/data/test_y_1.0.0_continuous.csv")

# Load counterfactuals
train_potential = pd.read_csv("./student_performance_cate_results/train_potential_y_1.0.0_continuous_cate.csv")
train_cf_cate = []
for i in range(len(train_potential)):
    if train_t.iloc[i, 0] == 0:
        train_cf_cate.append(train_potential.iloc[i, 1])
    else:
        train_cf_cate.append(train_potential.iloc[i, 0])
train_cf_cate = pd.DataFrame(train_cf_cate, columns=['0'])

# Calculate probability constants (same as fair_model.py)
probunos = train_t.iloc[:, 0].sum() / len(train_t)
probceros = (len(train_t) - train_t.iloc[:, 0].sum()) / len(train_t)

fact_const = []
coun_const = []
for i in train_t.iloc[:, 0]:
    if i == 1:
        fact_const.append(probunos)
        coun_const.append(probceros)
    else:
        fact_const.append(probceros)
        coun_const.append(probunos)

print("=== PROBABILITY ANALYSIS ===")
print(f"Treatment prevalence: {probunos:.4f}")
print(f"Control prevalence: {probceros:.4f}")
print(f"Fact_const mean: {np.mean(fact_const):.4f}")
print(f"Coun_const mean: {np.mean(coun_const):.4f}")

# Test different alpha values
alphas = [0.0, 0.5, 1.0]
for a_f in alphas:
    # Probability-weighted combination (from fair_model.py)
    train_y_combined = a_f * (train_y.iloc[:, 0]) + (1 - a_f) * ((train_cf_cate.iloc[:, 0] * coun_const + train_y.iloc[:, 0] * fact_const))
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y_combined)
    pred = rf.predict(test_x)
    mse = mean_squared_error(test_y, pred)
    
    print(f"\nAlpha {a_f:.1f}:")
    print(f"  Combined mean: {train_y_combined.mean():.4f}")
    print(f"  MSE: {mse:.4f}")

# Check what happens at α=0
print(f"\n=== ALPHA=0 BREAKDOWN ===")
weighted_cf = train_cf_cate.iloc[:, 0] * coun_const + train_y.iloc[:, 0] * fact_const
print(f"Weighted combination mean: {weighted_cf.mean():.4f}")
print(f"Pure counterfactual mean: {train_cf_cate.iloc[:, 0].mean():.4f}")
print(f"Pure factual mean: {train_y.iloc[:, 0].mean():.4f}")