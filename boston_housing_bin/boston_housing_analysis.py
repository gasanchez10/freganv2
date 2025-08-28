import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from fairlearn.datasets import fetch_boston
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 80)

# Load Boston housing dataset
X, y = fetch_boston(return_X_y=True)
boston_housing = pd.concat([X, y], axis=1)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

# Convert target to binary classification problem
y_binary = (y > np.median(y)).astype(int)
print(f"\nTarget converted to binary: median = {np.median(y):.2f}")
print(f"Binary target distribution: {np.bincount(y_binary)}")

# Apply transformations as specified
X_clf = X.assign(B_binary=lambda d: d['B'] > 136.9,
                 LSTAT=lambda d: d['LSTAT'] > np.median(d['LSTAT']))
# Remove original continuous B column, keep binary version
X_clf = X_clf.drop(columns=['B'])

print(f"\nTransformed data shape: {X_clf.shape}")
print(f"Binary target range: {y_binary.min()} - {y_binary.max()}")
print(f"Binary B distribution: {X_clf['B_binary'].value_counts().to_dict()}")

# Data exploration
print("\n=== DATA TYPES ===")
print(X_clf.dtypes.value_counts())
print("\n=== MISSING VALUES ===")
print("Features:", X_clf.isnull().sum().sum())
print("Targets:", y_binary.sum())

# Train classification model
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_binary, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Doubly Robust ATE Estimation
def doubly_robust_ate(X, y, treatment_col='B_binary'):
    """
    Estimate Average Treatment Effect using Doubly Robust estimation.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Treatment is binary B column
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Step 1: Estimate propensity scores P(T=1|X)
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X) - using classification for binary outcome
    mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
    mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    
    # Step 3: Doubly Robust ATE estimation
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    
    # Regression-only ATE for comparison
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'propensity_scores': ps,
        'treated_fraction': np.mean(T)
    }

def doubly_robust_cate(X, y, treatment_col='B_binary'):
    """
    Estimate Conditional Average Treatment Effect using Doubly Robust estimation.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
    mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    
    # Individual-level doubly robust CATE
    cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    
    # Regression-only CATE for comparison
    cate_reg = mu1 - mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression_only': cate_reg,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

# Run Doubly Robust ATE estimation
print("\n" + "="*50)
print("DOUBLY ROBUST ATE ESTIMATION")
print("="*50)

dr_results = doubly_robust_ate(X_clf, y_binary)

print(f"Treatment: B > 136.9 (proportion of blacks by town)")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")
print(f"\nInterpretation: B > 136.9 effect on high housing price probability is {dr_results['ate_doubly_robust']:.4f} (on average)")

# Run CATE estimation
print("\n" + "="*50)
print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
print("="*50)

cate_results = doubly_robust_cate(X_clf, y_binary)

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {np.mean(cate_results['cate_doubly_robust']):.4f}")
print(f"Mean CATE (Reg): {np.mean(cate_results['cate_regression_only']):.4f}")
print(f"Std CATE (Reg): {np.std(cate_results['cate_regression_only']):.4f}")

# Create comprehensive visualization (similar to synthetic_class_data)
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('Boston Housing Classification Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax1.figure.colorbar(im, ax=ax1)
ax1.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=['Low Price', 'High Price'],
        yticklabels=['Low Price', 'High Price'],
        title=f'Confusion Matrix\nAccuracy: {accuracy:.4f}',
        ylabel='True label',
        xlabel='Predicted label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Plot 2: Prediction Probabilities
ax2.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Low Price', bins=20, color='red')
ax2.hist(y_pred_proba[y_test == 1], alpha=0.7, label='High Price', bins=20, color='blue')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Probability Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Importance
importance = pd.Series(rf_model.feature_importances_, index=X_clf.columns).sort_values(ascending=False)
indices = np.argsort(rf_model.feature_importances_)[::-1]
ax3.bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_[indices], color='orange')
ax3.set_xlabel('Features')
ax3.set_ylabel('Importance')
ax3.set_title('Feature Importance')
ax3.set_xticks(range(len(rf_model.feature_importances_)))
ax3.set_xticklabels([X_clf.columns[i] for i in indices], rotation=45)
ax3.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Distribution (B_binary)
treatment_1 = y_test[X_test['B_binary'] == 1]
treatment_0 = y_test[X_test['B_binary'] == 0]
ax4.hist(treatment_0, alpha=0.7, label='B ≤ 136.9', bins=3, color='red', align='mid')
ax4.hist(treatment_1, alpha=0.7, label='B > 136.9', bins=3, color='blue', align='mid')
ax4.set_xlabel('Housing Price Category')
ax4.set_ylabel('Frequency')
ax4.set_title('Outcome Distribution by Treatment')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Low Price', 'High Price'])

# Plot 5: CATE Distribution
cate_estimates = cate_results['cate_regression_only']
ax5.hist(cate_estimates, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax5.set_xlabel('CATE Estimates')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of CATE Estimates')
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
    ax6.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('boston_housing_results.png', dpi=300, bbox_inches='tight')

print(f"\nTop 5 important features:")
print(importance.head().to_string())
print("\nMain results plot saved as 'boston_housing_results.png'")

# Additional ATE analysis plots
plt.figure(figsize=(20, 5))

# Plot 1: Housing Price Distribution by Treatment
plt.subplot(1, 4, 1)
T = X_clf['B_binary'].astype(int)
plt.hist(y_binary[T==0], alpha=0.7, label='B ≤ 136.9', bins=3, density=True)
plt.hist(y_binary[T==1], alpha=0.7, label='B > 136.9', bins=3, density=True)
plt.xlabel('Housing Price Category')
plt.ylabel('Density')
plt.title('Price Distribution by Treatment')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks([0, 1], ['Low Price', 'High Price'])

# Plot 2: Propensity Score Distribution
plt.subplot(1, 4, 2)
T = X_clf['B_binary'].astype(int)
ps = dr_results['propensity_scores']
plt.hist(ps[T==0], alpha=0.5, label='B ≤ 136.9 (Control)', bins=20, density=True)
plt.hist(ps[T==1], alpha=0.5, label='B > 136.9 (Treated)', bins=20, density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Outcome by Treatment Status
plt.subplot(1, 4, 3)
y_control = y_binary[T==0]
y_treated = y_binary[T==1]
plt.boxplot([y_control, y_treated], labels=['B ≤ 136.9', 'B > 136.9'])
plt.ylabel('Housing Price Category')
plt.title(f'Outcome by Treatment\nATE: {dr_results["ate_doubly_robust"]:.3f}')
plt.grid(True, alpha=0.3)
plt.yticks([0, 1], ['Low Price', 'High Price'])

# Plot 4: Treatment Effect Heterogeneity by Key Variable
plt.subplot(1, 4, 4)
# Show heterogeneity by LSTAT (socioeconomic status)
lstat_median = X_clf['LSTAT'].median()
high_lstat = X_clf['LSTAT'] > lstat_median

# Calculate ATE for each subgroup
ate_high_lstat = np.mean(y_binary[T==1][high_lstat[T==1]]) - np.mean(y_binary[T==0][high_lstat[T==0]])
ate_low_lstat = np.mean(y_binary[T==1][~high_lstat[T==1]]) - np.mean(y_binary[T==0][~high_lstat[T==0]])

subgroups = ['High LSTAT', 'Low LSTAT']
ate_values = [ate_high_lstat, ate_low_lstat]
plt.bar(subgroups, ate_values, alpha=0.7)
plt.ylabel('Average Treatment Effect')
plt.title('ATE Heterogeneity by LSTAT')
plt.grid(True, alpha=0.3)
for i, v in enumerate(ate_values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('boston_housing_ate_analysis.png', dpi=300, bbox_inches='tight')
print("\nATE analysis plots saved as 'boston_housing_ate_analysis.png'")

# Additional CATE visualization
plt.figure(figsize=(12, 4))

# Plot CATE distribution
plt.subplot(1, 2, 1)
cate_individual = cate_results['cate_regression_only']
plt.hist(cate_individual, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Individual Treatment Effect')
plt.ylabel('Frequency')
plt.title(f'CATE Distribution\nMean: {np.mean(cate_individual):.3f}, Std: {np.std(cate_individual):.3f}')
plt.grid(True, alpha=0.3)

# Plot CATE vs key covariate
plt.subplot(1, 2, 2)
plt.scatter(X_clf['LSTAT'], cate_individual, alpha=0.6)
plt.xlabel('LSTAT (% lower status population)')
plt.ylabel('Individual Treatment Effect')
plt.title('CATE vs LSTAT')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boston_housing_cate_analysis.png', dpi=300, bbox_inches='tight')
print("CATE analysis plots saved as 'boston_housing_cate_analysis.png'")

# Export data for counterfactual analysis
print("\n" + "="*50)
print("EXPORTING DATA FOR COUNTERFACTUAL ANALYSIS")
print("="*50)

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Use B_binary as treatment variable
treatment_col = 'B_binary'
train_t = X_train[treatment_col].values.reshape(-1, 1)
test_t = X_test[treatment_col].values.reshape(-1, 1)

# Features without treatment
train_x = X_train.drop(columns=[treatment_col]).values
test_x = X_test.drop(columns=[treatment_col]).values

# Outcomes (binary)
train_y = y_train.values.reshape(-1, 1)
test_y = y_test.values.reshape(-1, 1)

# Save to CSV files
version = "1.0.0"
data_files = {
    'train_x': train_x,
    'train_t': train_t,
    'train_y': train_y,
    'test_x': test_x,
    'test_t': test_t,
    'test_y': test_y
}

for name, data in data_files.items():
    df = pd.DataFrame(data)
    filepath = os.path.join(data_dir, f"{name}_{version}_binary.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {name}: {data.shape} -> {filepath}")

print(f"\nData exported successfully to '{data_dir}/' directory")
print(f"Treatment variable: {treatment_col} (B > 136.9)")
print(f"Outcome variable: housing prices (binary classification)")
print(f"Features: {train_x.shape[1]} variables (excluding treatment)")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: Boston Housing (n={len(X_clf)})")
print(f"Treatment: B > 136.9 (proportion of blacks > 136.9)")
print(f"Outcome: Housing prices (binary: high/low price)")
print(f"Model Performance: Accuracy = {accuracy:.4f}")
print(f"Average Treatment Effect: {dr_results['ate_doubly_robust']:.4f}")
print(f"Treatment prevalence: {dr_results['treated_fraction']:.1%}")
print("\nFiles generated:")
print("- boston_housing_results.png (main analysis)")
print("- boston_housing_ate_analysis.png (ATE plots)")
print("- boston_housing_cate_analysis.png (CATE plots)")
print("- data/ folder with CSV files for counterfactual analysis")