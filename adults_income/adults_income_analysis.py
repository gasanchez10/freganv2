import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fairlearn.datasets import fetch_acs_income

# Fetch dataset
data = fetch_acs_income()
X = data.data
y = data.target
print("Available columns:", X.columns.tolist())

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

# Data exploration
print("\n=== DATA TYPES ===")
print(X.dtypes.value_counts())
print("\n=== MISSING VALUES ===")
print("Features:", X.isnull().sum().sum())
print("Targets:", y.isnull().sum().sum())

print(f"\nTarget (Income) statistics:")
print(f"Mean: {y.mean():.2f}")
print(f"Median: {y.median():.2f}")
print(f"Range: {y.min():.2f} - {y.max():.2f}")

# Convert to binary classification: y>50000 -> 1, else 0
y_binary = (y > 50000).astype(int)
print(f"\nBinary classification target:")
print(f"High income (>$50k): {y_binary.sum()} ({100*y_binary.mean():.1f}%)")
print(f"Low income (≤$50k): {(1-y_binary).sum()} ({100*(1-y_binary.mean()):.1f}%)")

# Handle categorical variables
X_clean = X.copy()
categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    X_clean[col] = LabelEncoder().fit_transform(X_clean[col].astype(str))

# Fill missing numerical values
X_clean = X_clean.fillna(X_clean.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)

# Train classification model with soft parameters for large dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print(f"\nClassification Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['≤$50k', '>$50k'])
plt.yticks(tick_marks, ['≤$50k', '>$50k'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

plt.subplot(1, 3, 2)
plt.hist(y_pred_proba[y_test==0], alpha=0.7, label='≤$50k', bins=30, density=True)
plt.hist(y_pred_proba[y_test==1], alpha=0.7, label='>$50k', bins=30, density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
importance = pd.Series(rf_model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
importance.head(10).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top 10 Features')

plt.tight_layout()
plt.savefig('adults_income_results.png', dpi=300, bbox_inches='tight')

print(f"\nTop 5 important features:")
print(importance.head().to_string())
print("\nPlot saved as 'adults_income_results.png'")

# Doubly Robust ATE Estimation for binary outcome
def doubly_robust_ate(X, y, treatment_col='SEX'):
    from sklearn.ensemble import RandomForestClassifier
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Step 1: Estimate propensity scores P(T=1|X)
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X) for binary outcome
    mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
    mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    
    # Step 3: Doubly Robust ATE estimation for binary outcome
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    
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

dr_results = doubly_robust_ate(X_scaled, y_binary)

print(f"Treatment: Gender (SEX variable)")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")
print(f"\nInterpretation: Gender difference in high income probability is {dr_results['ate_doubly_robust']:.4f} (on average)")

# Plot ATE analysis
plt.figure(figsize=(20, 5))

# Plot 1: Income Distribution by Treatment
plt.subplot(1, 4, 1)
T = X_scaled['SEX'].astype(int)
plt.hist(y_binary[T==0], alpha=0.7, label='Female', bins=2, density=True)
plt.hist(y_binary[T==1], alpha=0.7, label='Male', bins=2, density=True)
plt.xlabel('High Income (>$50k)')
plt.ylabel('Density')
plt.title('Income Distribution by Gender')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Propensity Score Distribution
plt.subplot(1, 4, 2)
ps = dr_results['propensity_scores']
plt.hist(ps[T==0], alpha=0.5, label='Female (Control)', bins=20, density=True)
plt.hist(ps[T==1], alpha=0.5, label='Male (Treated)', bins=20, density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Outcome by Treatment Status
plt.subplot(1, 4, 3)
y_control = y_binary[T==0]
y_treated = y_binary[T==1]
plt.bar(['Female', 'Male'], [y_control.mean(), y_treated.mean()], alpha=0.7)
plt.ylabel('High Income Rate')
plt.title(f'Outcome by Treatment\nATE: {dr_results["ate_doubly_robust"]:.3f}')
plt.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Heterogeneity by Key Variable
plt.subplot(1, 4, 4)
numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    key_var = numeric_cols[0] if numeric_cols[0] != 'SEX' else numeric_cols[1]
    var_median = X_scaled[key_var].median()
    high_var = X_scaled[key_var] > var_median
    
    # Calculate ATE for each subgroup
    ate_high = np.mean(y_binary[T==1][high_var[T==1]]) - np.mean(y_binary[T==0][high_var[T==0]])
    ate_low = np.mean(y_binary[T==1][~high_var[T==1]]) - np.mean(y_binary[T==0][~high_var[T==0]])
    
    subgroups = [f'High {key_var}', f'Low {key_var}']
    ate_values = [ate_high, ate_low]
    plt.bar(subgroups, ate_values, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title(f'ATE Heterogeneity by {key_var}')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(ate_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('adults_income_ate_analysis.png', dpi=300, bbox_inches='tight')
print("\nATE analysis plots saved as 'adults_income_ate_analysis.png'")

def doubly_robust_cate(X, y, treatment_col='SEX'):
    from sklearn.ensemble import RandomForestClassifier
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
    mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    
    # Individual CATE estimates using doubly robust method for binary outcome
    cate_dr = np.mean(((((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0)))**2))
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

cate_results = doubly_robust_cate(X_scaled, y_binary)

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {cate_results['cate_doubly_robust']:.4f}")
print(f"Mean CATE (Reg): {np.mean(cate_results['cate_regression_only']):.4f}")
print(f"Std CATE (Reg): {np.std(cate_results['cate_regression_only']):.4f}")

# Show heterogeneity by key variables
numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
for var in numeric_cols[:3]:  # Check first 3 numeric variables
    if var != 'SEX' and var in X_scaled.columns:
        var_values = X_scaled[var]
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
    key_var = numeric_cols[0] if numeric_cols[0] != 'SEX' else numeric_cols[1]
    plt.scatter(X_scaled[key_var], cate_individual, alpha=0.6)
    plt.xlabel(f'{key_var}')
    plt.ylabel('Individual Treatment Effect')
    plt.title(f'CATE vs {key_var}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adults_income_cate_analysis.png', dpi=300, bbox_inches='tight')
print("CATE analysis plots saved as 'adults_income_cate_analysis.png'")

# Export data for counterfactual analysis
print("\n" + "="*50)
print("EXPORTING DATA FOR COUNTERFACTUAL ANALYSIS")
print("="*50)

# Create data directory if it doesn't exist
import os
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Use SEX as treatment variable
treatment_col = 'SEX'
train_t = X_train[treatment_col].values.reshape(-1, 1)
test_t = X_test[treatment_col].values.reshape(-1, 1)

# Features without treatment
train_x = X_train.drop(columns=[treatment_col]).values
test_x = X_test.drop(columns=[treatment_col]).values

# Binary outcomes
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
    filename = f"{data_dir}/{name}_{version}_continuous.csv"
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"Saved {filename} with shape {data.shape}")

print(f"\nData exported successfully to {data_dir}/ directory")
print(f"Treatment variable: {treatment_col} (SEX)")
print(f"Outcome variable: High income (>$50k) - binary classification")
print(f"Features: {train_x.shape[1]} variables (excluding treatment)")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: Adults Income (n={len(X_scaled)})")
print(f"Treatment: Gender (SEX)")
print(f"Outcome: High income (>$50k) - binary")
print(f"Model Performance: Accuracy = {accuracy:.4f}")
print(f"Average Treatment Effect: {dr_results['ate_doubly_robust']:.4f}")
print(f"Treatment prevalence: {dr_results['treated_fraction']:.1%}")
print("\nFiles generated:")
print("- adults_income_results.png (main analysis)")
print("- adults_income_ate_analysis.png (ATE plots)")
print("- adults_income_cate_analysis.png (CATE plots)")
print("- data/ folder with CSV files for counterfactual analysis")