import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from ucimlrepo import fetch_ucirepo

# Fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

print("Original dataset shape:", X.shape)

# Extract gender from personal status (Attribute9)
def extract_gender(personal_status):
    if personal_status in ['A91', 'A93', 'A94']:
        return 1  # male
    elif personal_status in ['A92']:
        return 0  # female
    else:
        return -1  # unknown

X['sex'] = X['Attribute9'].apply(extract_gender)

# Convert target to binary (1 for good credit, 0 for bad credit)
y_binary = (y['class'] == 1).astype(int)

# Remove rows with unknown gender
mask = X['sex'] != -1
X_clean = X[mask].copy()
y_clean = y_binary[mask].copy()

print("Available variables:", list(X_clean.columns))

# SELECT ONLY CONFOUNDER VARIABLES for German credit
# Key confounders that affect both treatment (gender) and outcome (credit approval):
confounder_vars = [
    'sex',              # Treatment variable (keep)
    'Attribute1',       # Checking account status - financial behavior
    'Attribute2',       # Duration of credit - loan characteristics
    'Attribute3',       # Credit history - past financial behavior
    'Attribute4',       # Purpose of credit - spending patterns
    'Attribute5',       # Credit amount - loan size
    'Attribute6',       # Savings account/bonds - financial assets
    'Attribute7',       # Present employment since - job stability
    'Attribute8',       # Installment rate - payment capacity
    'Attribute10',      # Other debtors/guarantors - financial support
    'Attribute11',      # Present residence since - stability
    'Attribute12',      # Property - assets/collateral
    'Attribute13',      # Age - life stage affects both gender patterns and creditworthiness
    'Attribute14',      # Other installment plans - existing obligations
    'Attribute15',      # Housing - living situation/stability
    'Attribute16',      # Number of existing credits - credit exposure
    'Attribute17',      # Job - employment type/income
    'Attribute18',      # Number of dependents - financial obligations
    'Attribute20'       # Foreign worker - residency status
]

# Filter to only available confounders
available_confounders = [var for var in confounder_vars if var in X_clean.columns]
print(f"\nSelected confounders ({len(available_confounders)}): {available_confounders}")

# Create filtered dataset with only confounders
X_confounders = X_clean[available_confounders].copy()

print(f"Filtered dataset shape: {X_confounders.shape}")
print(f"Reduction: {X_clean.shape[1]} -> {X_confounders.shape[1]} variables")

# Encode categorical variables
categorical_cols = X_confounders.select_dtypes(include=['object']).columns.tolist()
X_encoded = X_confounders.copy()

label_encoders = {}
for col in categorical_cols:
    if col != 'sex':  # sex is already encoded
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

# Train model on confounders only
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_clean, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance (Confounders Only):")
print(f"Accuracy: {accuracy:.4f}")

# Doubly Robust ATE with confounders only
def doubly_robust_ate(X, y, treatment_col='sex'):
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T)
    if ps_model.classes_.shape[0] == 1:
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
    
    if len(X_features[T==0]) > 0 and len(X_features[T==1]) > 0:
        mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
        mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    else:
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
    
    # Doubly Robust ATE
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(T)
    }

# Calculate ATE with confounders only
dr_results = doubly_robust_ate(X_encoded, y_clean)

print(f"\nCausal Analysis (Confounders Only):")
print(f"ATE (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"ATE (Regression): {dr_results['ate_regression_only']:.4f}")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")

# Export filtered data for counterfactual analysis
print(f"\nExporting confounders-only data...")

data_dir = "data_confounders"
os.makedirs(data_dir, exist_ok=True)

# Use sex as treatment variable (invert if ATE is negative)
treatment_col = 'sex'
if dr_results['ate_doubly_robust'] < 0:
    print("ATE is negative, inverting treatment for positive effect...")
    train_t = (1 - X_train[treatment_col]).values.reshape(-1, 1)
    test_t = (1 - X_test[treatment_col]).values.reshape(-1, 1)
    treatment_name = "Gender (Female)"
else:
    train_t = X_train[treatment_col].values.reshape(-1, 1)
    test_t = X_test[treatment_col].values.reshape(-1, 1)
    treatment_name = "Gender (Male)"

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
print(f"Treatment: {treatment_name}")
print(f"Features: {train_x.shape[1]} confounder variables")
print(f"Excluded: Non-confounder variables and irrelevant attributes")