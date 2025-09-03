import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# Fetch dataset
student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features
y = student_performance.data.targets

print("Original dataset shape:", X.shape)

# Clean data
X_clean = X.copy()
y_clean = y['G3'].dropna()
X_clean = X_clean.loc[y_clean.index]

# Handle categorical variables
categorical_cols = X_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X_clean[col] = LabelEncoder().fit_transform(X_clean[col].astype(str))

# Fill missing numerical values
X_clean = X_clean.fillna(X_clean.median())

print("Available variables:", list(X_clean.columns))

# SELECT ONLY CONFOUNDER VARIABLES (exclude omitted/repetitive variables)
# Key confounders that affect both treatment (sex) and outcome (G3):
confounder_vars = [
    'sex',        # Treatment variable (keep)
    'age',        # Demographics - affects both gender patterns and performance
    'Medu',       # Mother's education - key confounder
    'Fedu',       # Father's education - key confounder  
    'studytime',  # Study habits - affects performance
    'failures',   # Past failures - strong predictor
    'schoolsup',  # School support - affects performance
    'famsup',     # Family support - affects performance
    'paid',       # Extra paid classes - affects performance
    'activities', # Extra activities - time allocation
    'higher',     # Wants higher education - motivation
    'internet',   # Internet access - resources
    'romantic',   # Romantic relationship - time/focus
    'famrel',     # Family relationships - support system
    'freetime',   # Free time - study balance
    'goout',      # Going out - social vs study time
    'Dalc',       # Workday alcohol - behavior
    'Walc',       # Weekend alcohol - behavior
    'health',     # Health status - affects performance
    'absences'    # School absences - engagement
]

# Filter to only available confounders
available_confounders = [var for var in confounder_vars if var in X_clean.columns]
print(f"\nSelected confounders ({len(available_confounders)}): {available_confounders}")

# Create filtered dataset with only confounders
X_confounders = X_clean[available_confounders].copy()

print(f"Filtered dataset shape: {X_confounders.shape}")
print(f"Reduction: {X_clean.shape[1]} -> {X_confounders.shape[1]} variables")

# Train model on confounders only
X_train, X_test, y_train, y_test = train_test_split(X_confounders, y_clean, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate MSE
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nModel Performance (Confounders Only):")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Doubly Robust ATE with confounders only
def doubly_robust_ate(X, y, treatment_col='sex'):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Doubly Robust ATE
    ate_dr = (np.mean(T*(y - mu1)/ps + mu1) - np.mean((1-T)*(y - mu0)/(1-ps) + mu0))
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(T)
    }

# Calculate ATE with confounders only
dr_results = doubly_robust_ate(X_confounders, y_clean)

print(f"\nCausal Analysis (Confounders Only):")
print(f"ATE (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"ATE (Regression): {dr_results['ate_regression_only']:.4f}")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")

# Export filtered data for counterfactual analysis
print(f"\nExporting confounders-only data...")

data_dir = "data_confounders"
os.makedirs(data_dir, exist_ok=True)

# Use sex as treatment variable (inverted for positive ATE)
treatment_col = 'sex'
train_t = (1 - X_train[treatment_col]).values.reshape(-1, 1)  # Invert treatment
test_t = (1 - X_test[treatment_col]).values.reshape(-1, 1)   # Invert treatment

# Features without treatment
train_x = X_train.drop(columns=[treatment_col]).values
test_x = X_test.drop(columns=[treatment_col]).values

# Outcomes
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

print(f"\nConfounders-only data exported to '{data_dir}/' directory")
print(f"Treatment: Inverted sex (0=male, 1=female) for positive ATE")
print(f"Features: {train_x.shape[1]} confounder variables")
print(f"Excluded: Omitted variables, repetitive variables, and non-confounders")