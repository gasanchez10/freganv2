# CATE Counterfactuals Analysis for German Credit (Binary Classification)
# This script generates counterfactual outcomes using CATE estimation methods

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')  # Go up one level to access data
results_dir = os.path.join(current_dir, 'cate_results')
os.makedirs(results_dir, exist_ok=True)

version = "1.0.0"
np.random.seed(123)

print("=== CATE COUNTERFACTUALS ANALYSIS FOR GERMAN CREDIT (BINARY) ===")

# Load data
print("Loading German credit data...")
train_x = pd.read_csv(os.path.join(data_dir, f'train_x_{version}_binary.csv'))
train_t = pd.read_csv(os.path.join(data_dir, f'train_t_{version}_binary.csv')).values.flatten()
train_y = pd.read_csv(os.path.join(data_dir, f'train_y_{version}_binary.csv')).values.flatten()

test_x = pd.read_csv(os.path.join(data_dir, f'test_x_{version}_binary.csv'))
test_t = pd.read_csv(os.path.join(data_dir, f'test_t_{version}_binary.csv')).values.flatten()
test_y = pd.read_csv(os.path.join(data_dir, f'test_y_{version}_binary.csv')).values.flatten()

# Prepare full dataset
full_x = np.vstack([train_x, test_x])
full_t = np.concatenate([train_t, test_t])
full_y = np.concatenate([train_y, test_y])

print(f"Dataset size: {len(full_x)} samples")
print(f"Treatment distribution: {np.mean(full_t):.3f} treated")

def doubly_robust_cate(X, T, Y):
    """Calculate CATE using doubly robust estimation for binary outcomes."""
    # Propensity score estimation with clipping to avoid division by zero
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, T).predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)  # Clip to avoid division by zero
    
    # Outcome models for T=0 and T=1
    mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[T==0], Y[T==0]).predict_proba(X)[:, 1]
    mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[T==1], Y[T==1]).predict_proba(X)[:, 1]
    
    # Doubly robust CATE with safe division
    cate = (T*(Y - mu1)/ps + mu1) - ((1-T)*(Y - mu0)/(1-ps) + mu0)
    
    return cate, {'mu0': mu0, 'mu1': mu1, 'ps': ps}

# Generate counterfactuals using doubly robust CATE
print(f"\nEstimating CATE using doubly robust method...")

cate_estimates, components = doubly_robust_cate(full_x, full_t, full_y)

# Generate counterfactual outcomes as probabilities
counterfactuals = np.where(full_t == 1, 
                          full_y - cate_estimates,
                          full_y + cate_estimates)

# Clip to probability range [0,1] but keep as probabilities
counterfactuals_probs = np.clip(counterfactuals, 0, 1)

print(f"  Mean CATE: {np.mean(cate_estimates):.4f}")
print(f"  Std CATE: {np.std(cate_estimates):.4f}")

# Create potential outcomes structure
train_potential_y = []
test_potential_y = []

# Split counterfactuals back to train/test (use probabilities)
train_cf_cate = counterfactuals_probs[:len(train_y)]
test_cf_cate = counterfactuals_probs[len(train_y):]

for i in range(len(train_y)):
    if train_t[i] == 0:  # Control (female)
        train_potential_y.append([train_y[i], train_cf_cate[i]])  # [Y0, Y1]
    else:  # Treatment (male)
        train_potential_y.append([train_cf_cate[i], train_y[i]])  # [Y0, Y1]

for i in range(len(test_y)):
    if test_t[i] == 0:  # Control (female)
        test_potential_y.append([test_y[i], test_cf_cate[i]])  # [Y0, Y1]
    else:  # Treatment (male)
        test_potential_y.append([test_cf_cate[i], test_y[i]])  # [Y0, Y1]

train_potential_y = np.array(train_potential_y)
test_potential_y = np.array(test_potential_y)
all_potential_y = np.vstack([train_potential_y, test_potential_y])

# Save results
print(f"\nSaving results to {results_dir}...")

# Save all data arrays
data_arrays = [
    train_x, train_t.reshape(-1, 1), train_y.reshape(-1, 1), train_potential_y,
    test_x, test_t.reshape(-1, 1), test_y.reshape(-1, 1), test_potential_y, 
    all_potential_y
]
data_names = [
    "train_x", "train_t", "train_y", "train_potential_y",
    "test_x", "test_t", "test_y", "test_potential_y", 
    "test_y_hat"
]

for name, array in zip(data_names, data_arrays):
    df = pd.DataFrame(array)
    save_path = os.path.join(results_dir, f"{name}_{version}_binary_cate.csv")
    df.to_csv(save_path, index=False)

# Save CATE estimates
pd.DataFrame({'cate': cate_estimates}).to_csv(
    os.path.join(results_dir, 'cate_estimates.csv'), index=False)

# Save counterfactuals (as probabilities)
pd.DataFrame({'counterfactuals': counterfactuals_probs}).to_csv(
    os.path.join(results_dir, 'counterfactuals.csv'), index=False)

# Save components
components_df = pd.DataFrame(components)
components_df.to_csv(os.path.join(results_dir, 'components.csv'), index=False)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('CATE Counterfactuals Analysis - German Credit', fontsize=16, fontweight='bold')

# Plot 1: CATE distribution
ax1.hist(cate_estimates, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.set_title(f'CATE Distribution\nMean: {np.mean(cate_estimates):.3f}')
ax1.set_xlabel('CATE Estimates')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Plot 2: Factual vs Counterfactual correlation
correlation = np.corrcoef(full_y, counterfactuals_probs)[0, 1]
ax2.scatter(full_y, counterfactuals_probs, alpha=0.6, color='blue')
ax2.plot([0, 1], [0, 1], 'r--', lw=2)
ax2.set_title(f'Factual vs Counterfactual\nCorr: {correlation:.3f}')
ax2.set_xlabel('Factual Outcomes')
ax2.set_ylabel('Counterfactual Outcomes')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'cate_counterfactuals_analysis.png'), 
           dpi=300, bbox_inches='tight')
plt.close()

# Summary
print(f"\n{'='*50}")
print("SUMMARY RESULTS")
print(f"{'='*50}")
print(f"Correlation factual vs counterfactual: {correlation:.4f}")
print(f"Mean CATE: {np.mean(cate_estimates):.4f}")
print(f"Std CATE: {np.std(cate_estimates):.4f}")

# Save correlation results
with open(os.path.join(results_dir, "cate_correlation_results.txt"), "w") as f:
    f.write("CATE Counterfactuals Results:\n")
    f.write(f"Correlation Factual vs Counterfactual: {correlation:.4f}\n")
    f.write(f"Mean CATE: {np.mean(cate_estimates):.4f}\n")
    f.write(f"Std CATE: {np.std(cate_estimates):.4f}\n")

# Save summary
summary_df = pd.DataFrame({
    'correlation': [correlation],
    'mean_cate': [np.mean(cate_estimates)],
    'std_cate': [np.std(cate_estimates)]
})
summary_df.to_csv(os.path.join(results_dir, 'summary_results.csv'), index=False)

print(f"\nAnalysis complete! Results saved in {results_dir}")