import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

# Apply transformations as specified - keep target continuous
X_clf = X.assign(B_binary=lambda d: d['B'] > 136.9,
                 LSTAT=lambda d: d['LSTAT'] > np.median(d['LSTAT']))
# Remove original continuous B column, keep binary version
X_clf = X_clf.drop(columns=['B'])

print(f"\nTransformed data shape: {X_clf.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")
print(f"Binary B distribution: {X_clf['B_binary'].value_counts().to_dict()}")

# Data exploration
print("\n=== DATA TYPES ===")
print(X_clf.dtypes.value_counts())
print("\n=== MISSING VALUES ===")
print("Features:", X_clf.isnull().sum().sum())
print("Targets:", y.isnull().sum().sum())

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_clf, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate MSE
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Housing Price')
plt.ylabel('Predicted Housing Price')
plt.title(f'Actual vs Predicted\nMSE: {mse:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Housing Price')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
importance = pd.Series(rf_model.feature_importances_, index=X_clf.columns).sort_values(ascending=False)
importance.head(10).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top 10 Features')

plt.tight_layout()
plt.savefig('boston_housing_results.png', dpi=300, bbox_inches='tight')

print(f"\nTop 5 important features:")
print(importance.head().to_string())
print("\nMain results plot saved as 'boston_housing_results.png'")

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
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X) - using regression for continuous outcome
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Step 3: Doubly Robust ATE estimation
    ate_dr = (np.mean(T*(y - mu1)/ps + mu1) - 
              np.mean((1-T)*(y - mu0)/(1-ps) + mu0))
    
    # Regression-only ATE for comparison
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'propensity_scores': ps,
        'treated_fraction': np.mean(T)
    }

# Run Doubly Robust ATE estimation
print("\n" + "="*50)
print("DOUBLY ROBUST ATE ESTIMATION")
print("="*50)

dr_results = doubly_robust_ate(X_clf, y)

print(f"Treatment: B > 136.9 (proportion of blacks by town)")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")
print(f"\nInterpretation: B > 136.9 effect on housing prices is {dr_results['ate_doubly_robust']:.4f} thousand dollars (on average)")

# Plot ATE analysis
plt.figure(figsize=(20, 5))

# Plot 1: Housing Price Distribution by Treatment
plt.subplot(1, 4, 1)
T = X_clf['B_binary'].astype(int)
plt.hist(y[T==0], alpha=0.7, label='B ≤ 136.9', bins=20, density=True)
plt.hist(y[T==1], alpha=0.7, label='B > 136.9', bins=20, density=True)
plt.xlabel('Housing Price ($1000s)')
plt.ylabel('Density')
plt.title('Price Distribution by Treatment')
plt.legend()
plt.grid(True, alpha=0.3)

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
y_control = y[T==0]
y_treated = y[T==1]
plt.boxplot([y_control, y_treated], labels=['B ≤ 136.9', 'B > 136.9'])
plt.ylabel('Housing Price ($1000s)')
plt.title(f'Outcome by Treatment\nATE: {dr_results["ate_doubly_robust"]:.3f}')
plt.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Heterogeneity by Key Variable
plt.subplot(1, 4, 4)
# Show heterogeneity by LSTAT (socioeconomic status)
lstat_median = X_clf['LSTAT'].median()
high_lstat = X_clf['LSTAT'] > lstat_median

# Calculate ATE for each subgroup
ate_high_lstat = np.mean(y[T==1][high_lstat[T==1]]) - np.mean(y[T==0][high_lstat[T==0]])
ate_low_lstat = np.mean(y[T==1][~high_lstat[T==1]]) - np.mean(y[T==0][~high_lstat[T==0]])

subgroups = ['High LSTAT', 'Low LSTAT']
ate_values = [ate_high_lstat, ate_low_lstat]
plt.bar(subgroups, ate_values, alpha=0.7)
plt.ylabel('Average Treatment Effect')
plt.title('ATE Heterogeneity by LSTAT')
plt.grid(True, alpha=0.3)
for i, v in enumerate(ate_values):
    plt.text(i, v + 0.1, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('boston_housing_ate_analysis.png', dpi=300, bbox_inches='tight')
print("\nATE analysis plots saved as 'boston_housing_ate_analysis.png'")

# Conditional Average Treatment Effect (CATE) Estimation
def doubly_robust_cate(X, y, treatment_col='B_binary'):
    """
    Estimate Conditional Average Treatment Effect using Doubly Robust estimation.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Individual-level doubly robust CATE
    cate_dr = np.mean((((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))**2))
    
    # Regression-only CATE for comparison
    cate_reg = mu1 - mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression_only': cate_reg,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

# Run CATE estimation
print("\n" + "="*50)
print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
print("="*50)

cate_results = doubly_robust_cate(X_clf, y)

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {cate_results['cate_doubly_robust']:.4f}")
print(f"Mean CATE (Reg): {np.mean(cate_results['cate_regression_only']):.4f}")
print(f"Std CATE (Reg): {np.std(cate_results['cate_regression_only']):.4f}")

# Show heterogeneity by key variables
for var in ['LSTAT', 'RM', 'CRIM']:
    if var in X_clf.columns:
        var_values = X_clf[var]
        high_var = var_values > var_values.median()
        cate_high = np.mean(cate_results['cate_regression_only'][high_var])
        cate_low = np.mean(cate_results['cate_regression_only'][~high_var])
        print(f"CATE for high {var}: {cate_high:.4f}")
        print(f"CATE for low {var}: {cate_low:.4f}")
        print(f"Difference: {cate_high - cate_low:.4f}\n")

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

# Outcomes (continuous)
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
    filepath = os.path.join(data_dir, f"{name}_{version}_continuous.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {name}: {data.shape} -> {filepath}")

print(f"\nData exported successfully to '{data_dir}/' directory")
print(f"Treatment variable: {treatment_col} (B > 136.9)")
print(f"Outcome variable: housing prices (continuous)")
print(f"Features: {train_x.shape[1]} variables (excluding treatment)")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: Boston Housing (n={len(X_clf)})")
print(f"Treatment: B > 136.9 (proportion of blacks > 136.9)")
print(f"Outcome: Housing prices (continuous, $1000s)")
print(f"Model Performance: MSE = {mse:.4f}, RMSE = {rmse:.4f}")
print(f"Average Treatment Effect: {dr_results['ate_doubly_robust']:.4f}")
print(f"Treatment prevalence: {dr_results['treated_fraction']:.1%}")
print("\nFiles generated:")
print("- boston_housing_results.png (main analysis)")
print("- boston_housing_ate_analysis.png (ATE plots)")
print("- boston_housing_cate_analysis.png (CATE plots)")
print("- data/ folder with CSV files for counterfactual analysis")