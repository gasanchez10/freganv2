# Synthetic Data Analysis with Random Forest Regression
# This script generates synthetic causal data and applies Random Forest regression

import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.datasets import make_regression

# Configuration parameters
version = "1.0.0"
N = 1000  # Sample size
mu = 0    # Mean for normal distributions
sigma = 3 # Standard deviation
np.random.seed(123)  # Reproducibility
train_rate = 0.7     # Training set proportion
current_dir = os.path.dirname(os.path.realpath(__file__))

# Sigmoid activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Generate exogenous variables (confounders)
Z1 = np.random.normal(mu, sigma, N) 
Z2 = np.random.normal(mu, sigma, N) 

# Generate treatment assignment based on confounders
p = sigmoid(2*Z1 - 1*Z2)
T = [np.random.binomial(1, i, [1,1]) for i in p]
T = np.array(T).reshape([N,])

# Generate unobserved error terms
UJ = np.random.normal(mu, sigma, N)
UP = np.random.normal(mu, sigma, N)
UD = np.random.normal(mu, sigma, N)
U = np.random.normal(mu, sigma, N)

# Generate mediator variables following structural equations
J = 3*T - 2*Z1 + 3*Z2 + UJ
P = J - 2*T + 2*Z2 + 0.5*Z1 + UP
D = 2*J + P + 2*T + Z1 + Z2 + UD

# Generate outcome variable with strong positive treatment effect but realistic CATE
Y = 20*T + 1*Z1 + 1*Z2 + 0.3*J + 0.2*D + U

# Create factual dataset
df = pd.DataFrame(data=[T, Z1, Z2, J, P, D, Y]).T
df.columns = ["T", "Z1", "Z2", "J", "P", "D", "Y"]

# Generate counterfactuals by inverting treatment
df['U'] = U
T_CF = 1 - T  # Flip treatment

# Recalculate mediators under counterfactual treatment
J = 3*T_CF - 2*Z1 + 3*Z2 + UJ
P = J - 2*T_CF + 2*Z2 + 0.5*Z1 + UP
D = 2*J + P + 2*T_CF + Z1 + Z2 + UD

# Generate counterfactual outcomes
Y_cf_Y = Y  # Baseline (no change)
Y_cf_Manual = 20*T_CF + 1*Z1 + 1*Z2 + 0.3*J + 0.2*D + U  # True counterfactual
Y_cf_Random = np.random.normal(mu, 100, N)  # Random baseline

df['Y_cf_Y'] = Y_cf_Y
df['Y_cf_Manual'] = Y_cf_Manual
df['Y_cf_Random'] = Y_cf_Random
# Generate dataframe output
# df.to_csv('output.csv', index=False)

# Prepare features and split data
X = np.array([Z1, Z2, J, P, D]).T
idx = np.random.permutation(N)
train_idx = idx[:int(train_rate * N)]
test_idx = idx[int(train_rate * N):]

# Create training set
train_x = np.array(X)[train_idx]
train_t = T[train_idx]
train_y = np.array(Y)[train_idx]
train_cf_y = np.array(Y_cf_Y)[train_idx]
train_cf_manual = np.array(Y_cf_Manual)[train_idx]
train_cf_random = np.array(Y_cf_Random)[train_idx]

# Create test set
test_x = np.array(X)[test_idx]
test_t = T[test_idx]
test_y = np.array(Y)[test_idx]
test_cf_y = np.array(Y_cf_Y)[test_idx]
test_cf_manual = np.array(Y_cf_Manual)[test_idx]
test_cf_random = np.array(Y_cf_Random)[test_idx]

# Save all datasets to CSV files
names = ["train_x", "train_t", "train_y", "train_cf_y", "train_cf_manual", "train_cf_random", 
         "test_x", "test_t", "test_y", "test_cf_y", "test_cf_manual", "test_cf_random"]
arr = [train_x, train_t, train_y, train_cf_y, train_cf_manual, train_cf_random, 
       test_x, test_t, test_y, test_cf_y, test_cf_manual, test_cf_random]

for i in range(len(arr)):
    aux = pd.DataFrame(arr[i])
    print("Generating:", names[i])
    save_path = os.path.join(current_dir, 'data', names[i]+'_'+version+'_continuous.csv')
    aux.to_csv(save_path, index=False)

# Prepare data for Random Forest regression
train_x_df = pd.DataFrame(train_x, columns=['Z1', 'Z2', 'J', 'P', 'D'])
train_x_df["T"] = train_t
test_x_df = pd.DataFrame(test_x, columns=['Z1', 'Z2', 'J', 'P', 'D'])
test_x_df["T"] = test_t

# Train Random Forest model
print("\n=== SYNTHETIC DATA ANALYSIS ===")
print("Training Random Forest Regression Model...")

rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf.fit(train_x_df, train_y)

# Make predictions
y_pred = rf.predict(test_x_df)

# Calculate metrics
mse = mean_squared_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Doubly Robust ATE Estimation (following same logic as student_performance)
def doubly_robust_ate(X, y, treatment_col='T'):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Step 1: Estimate propensity scores P(T=1|X)
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X)
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Step 3: Doubly Robust ATE estimation
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) -  ((1-T)*(y - mu0)/(1-ps) + mu0))
    
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'propensity_scores': ps,
        'treated_fraction': np.mean(T)
    }

def doubly_robust_cate(X, y, treatment_col='T'):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Individual CATE estimates using doubly robust method
    cate_dr = ((((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))))
    cate_reg = mu1 - mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression_only': cate_reg,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

# Prepare full dataset for causal analysis
full_X = pd.DataFrame(np.vstack([train_x_df, test_x_df]), 
                      columns=['Z1', 'Z2', 'J', 'P', 'D', 'T'])
full_y = np.concatenate([train_y, test_y])

print("\n" + "="*50)
print("DOUBLY ROBUST ATE ESTIMATION")
print("="*50)

dr_results = doubly_robust_ate(full_X, full_y)

print(f"Treatment: T variable")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")

print("\n" + "="*50)
print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
print("="*50)

cate_results = doubly_robust_cate(full_X, full_y)

cate_estimates = cate_results['cate_doubly_robust']

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {np.mean(cate_estimates):.4f}")
print(f"Std CATE (DR): {np.std(cate_estimates):.4f}")
print(f"Min CATE (DR): {np.min(cate_estimates):.4f}")
print(f"Max CATE (DR): {np.max(cate_estimates):.4f}")
model_ate = dr_results['ate_doubly_robust']

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('Synthetic Data Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1.scatter(test_y, y_pred, alpha=0.6, color='blue')
ax1.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Y')
ax1.set_ylabel('Predicted Y')
ax1.set_title(f'Actual vs Predicted\nMSE: {mse:.4f}, R²: {r2:.4f}')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = test_y.flatten() - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Y')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot')
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Importance
feature_names = ['Z1', 'Z2', 'J', 'P', 'D', 'T']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
ax3.bar(range(len(importances)), importances[indices], color='orange')
ax3.set_xlabel('Features')
ax3.set_ylabel('Importance')
ax3.set_title('Feature Importance')
ax3.set_xticks(range(len(importances)))
ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax3.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Distribution
treatment_1 = test_y[test_x_df['T'] == 1].flatten()
treatment_0 = test_y[test_x_df['T'] == 0].flatten()
ax4.hist(treatment_0, alpha=0.7, label='T=0', bins=20, color='red')
ax4.hist(treatment_1, alpha=0.7, label='T=1', bins=20, color='blue')
ax4.set_xlabel('Outcome Y')
ax4.set_ylabel('Frequency')
ax4.set_title('Outcome Distribution by Treatment')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: CATE Distribution
ax5.hist(cate_estimates, bins=30, alpha=0.7, color='purple', edgecolor='black')
#ax5.axvline(np.mean(cate_estimates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cate_estimates):.3f}')
print("CATE SQUARES MEAN: ", np.mean(cate_estimates**2))
ax5.set_xlabel('CATE Estimates')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of CATE Estimates')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: ATE Comparison
ate_methods = ['Doubly Robust', 'Regression Only']
ate_values = [dr_results['ate_doubly_robust'], dr_results['ate_regression_only']]
colors = ['green', 'blue']
ax6.bar(ate_methods, ate_values, color=colors, alpha=0.7)
ax6.set_ylabel('ATE Value')
ax6.set_title('ATE Estimation Methods')
ax6.grid(True, alpha=0.3)
for i, v in enumerate(ate_values):
    ax6.text(i, v + 0.1, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_path = os.path.join(current_dir, 'synthetic_data_results.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nResults saved to: {save_path}")
print("Analysis complete!")