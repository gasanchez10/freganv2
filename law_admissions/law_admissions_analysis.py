import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import kagglehub
from kagglehub import KaggleDatasetAdapter
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 80)

# Load law school admissions dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "danofer/law-school-admissions-bar-passage",
    "bar_pass_prediction.csv",
)

print("Dataset shape:", df.shape)
print("First 5 records:", df.head())

# Prepare features and target
y = df['lsat']

# Handle categorical variables - keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].drop(columns=['lsat'])

# Remove redundant gender/sex columns - keep only 'male' as binary treatment
redundant_cols = ['sex', 'gender'] if 'gender' in X.columns else ['sex']
X = X.drop(columns=[col for col in redundant_cols if col in X.columns])

# Remove other redundant race columns, keep 'black' as race treatment
race_cols_to_remove = ['race', 'race1', 'race2', 'asian', 'hisp', 'other']
X = X.drop(columns=[col for col in race_cols_to_remove if col in X.columns])

print(f"\nFeatures shape after removing redundant columns: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# Handle missing values
X = X.fillna(X.median())
y = y.fillna(y.median())

# Invert black treatment to get positive ATE (non-black as treatment)
X['non_black'] = (1 - X['black']).astype(int)
X = X.drop(columns=['black'])  # Remove original black column
print(f"Non-black distribution: {X['non_black'].value_counts().to_dict()}")
print(f"Male distribution: {X['male'].value_counts().to_dict()}")

# Data exploration
print("\n=== DATA TYPES ===")
print(X.dtypes.value_counts())
print("\n=== MISSING VALUES ===")
print("Features:", X.isnull().sum().sum())
print("Targets:", y.isnull().sum().sum())

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
plt.xlabel('Actual LSAT Score')
plt.ylabel('Predicted LSAT Score')
plt.title(f'Actual vs Predicted\nMSE: {mse:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted LSAT Score')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.head(10).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top 10 Features')

plt.tight_layout()
plt.savefig('law_admissions_results.png', dpi=300, bbox_inches='tight')

print(f"\nTop 5 important features:")
print(importance.head().to_string())
print("\nMain results plot saved as 'law_admissions_results.png'")

# Doubly Robust ATE Estimation
def doubly_robust_ate(X, y, treatment_col='non_black'):
    """
    Estimate Average Treatment Effect using Doubly Robust estimation.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Treatment is non_black column
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Step 1: Estimate propensity scores P(T=1|X)
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X) - using regression for continuous outcome
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Step 3: Doubly Robust ATE estimation
    ate_dr = np.mean(T*(y - mu1)/ps + mu1) - np.mean((1-T)*(y - mu0)/(1-ps) + mu0)
    
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

dr_results = doubly_robust_ate(X, y)

print(f"Treatment: Black (race)")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")
print(f"\nInterpretation: Black race effect on LSAT scores is {dr_results['ate_doubly_robust']:.4f} points (on average)")

# Plot ATE analysis
plt.figure(figsize=(20, 5))

# Plot 1: LSAT Score Distribution by Treatment
plt.subplot(1, 4, 1)
T = X['non_black'].astype(int)
plt.hist(y[T==0], alpha=0.7, label='Race=0', bins=20, density=True)
plt.hist(y[T==1], alpha=0.7, label='Race=1', bins=20, density=True)
plt.xlabel('LSAT Score')
plt.ylabel('Density')
plt.title('LSAT Distribution by Race')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Propensity Score Distribution
plt.subplot(1, 4, 2)
ps = dr_results['propensity_scores']
plt.hist(ps[T==0], alpha=0.5, label='Race=0 (Control)', bins=20, density=True)
plt.hist(ps[T==1], alpha=0.5, label='Race=1 (Treated)', bins=20, density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Outcome by Treatment Status
plt.subplot(1, 4, 3)
y_control = y[T==0]
y_treated = y[T==1]
plt.boxplot([y_control, y_treated], labels=['Race=0', 'Race=1'])
plt.ylabel('LSAT Score')
plt.title(f'Outcome by Treatment\nATE: {dr_results["ate_doubly_robust"]:.3f}')
plt.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Heterogeneity by Key Variable
plt.subplot(1, 4, 4)
# Show heterogeneity by first available numeric variable
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    key_var = numeric_cols[0] if numeric_cols[0] != 'race_binary' else numeric_cols[1]
    var_median = X[key_var].median()
    high_var = X[key_var] > var_median
    
    # Calculate ATE for each subgroup
    ate_high = np.mean(y[T==1][high_var[T==1]]) - np.mean(y[T==0][high_var[T==0]])
    ate_low = np.mean(y[T==1][~high_var[T==1]]) - np.mean(y[T==0][~high_var[T==0]])
    
    subgroups = [f'High {key_var}', f'Low {key_var}']
    ate_values = [ate_high, ate_low]
    plt.bar(subgroups, ate_values, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title(f'ATE Heterogeneity by {key_var}')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(ate_values):
        plt.text(i, v + 0.1, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('law_admissions_ate_analysis.png', dpi=300, bbox_inches='tight')
print("\nATE analysis plots saved as 'law_admissions_ate_analysis.png'")

# Conditional Average Treatment Effect (CATE) Estimation
def doubly_robust_cate(X, y, treatment_col='non_black'):
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

cate_results = doubly_robust_cate(X, y)

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {cate_results['cate_doubly_robust']:.4f}")
print(f"Mean CATE (Reg): {np.mean(cate_results['cate_regression_only']):.4f}")
print(f"Std CATE (Reg): {np.std(cate_results['cate_regression_only']):.4f}")

# Show heterogeneity by key variables
numeric_cols = X.select_dtypes(include=[np.number]).columns
for var in numeric_cols[:3]:  # Check first 3 numeric variables
    if var != 'race_binary' and var in X.columns:
        var_values = X[var]
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
if len(numeric_cols) > 1:
    key_var = numeric_cols[0] if numeric_cols[0] != 'race_binary' else numeric_cols[1]
    plt.scatter(X[key_var], cate_individual, alpha=0.6)
    plt.xlabel(f'{key_var}')
    plt.ylabel('Individual Treatment Effect')
    plt.title(f'CATE vs {key_var}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('law_admissions_cate_analysis.png', dpi=300, bbox_inches='tight')
print("CATE analysis plots saved as 'law_admissions_cate_analysis.png'")

# Export data for counterfactual analysis
print("\n" + "="*50)
print("EXPORTING DATA FOR COUNTERFACTUAL ANALYSIS")
print("="*50)

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Use non_black as treatment variable
treatment_col = 'non_black'
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
print(f"Treatment variable: {treatment_col} (race)")
print(f"Outcome variable: LSAT scores (continuous)")
print(f"Features: {train_x.shape[1]} variables (excluding treatment)")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: Law School Admissions (n={len(X)})")
print(f"Treatment: Black (race)")
print(f"Outcome: LSAT scores (continuous)")
print(f"Model Performance: MSE = {mse:.4f}, RMSE = {rmse:.4f}")
print(f"Average Treatment Effect: {dr_results['ate_doubly_robust']:.4f}")
print(f"Treatment prevalence: {dr_results['treated_fraction']:.1%}")
print("\nFiles generated:")
print("- law_admissions_results.png (main analysis)")
print("- law_admissions_ate_analysis.png (ATE plots)")
print("- law_admissions_cate_analysis.png (CATE plots)")
print("- data/ folder with CSV files for counterfactual analysis")