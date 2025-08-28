import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 80)

# Fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144)

# Data (as pandas dataframes)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

# Metadata and variable information
print("=== METADATA ===")
print(statlog_german_credit_data.metadata)
print("\n=== VARIABLE INFORMATION ===")
print(statlog_german_credit_data.variables)

print(f"\nDataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")
print(f"Target columns: {list(y.columns)}")

# Check personal status for gender information (Attribute9)
print("\n=== PERSONAL STATUS ANALYSIS ===")
print(f"Personal status unique values: {X['Attribute9'].unique()}")
print(f"Personal status value counts:")
print(X['Attribute9'].value_counts())

# Extract gender from personal status
# A91: male divorced/separated, A92: female divorced/separated/married
# A93: male single, A94: male married/widowed
def extract_gender(personal_status):
    if personal_status in ['A91', 'A93', 'A94']:
        return 1  # male
    elif personal_status in ['A92']:
        return 0  # female
    else:
        return -1  # unknown

X['sex'] = X['Attribute9'].apply(extract_gender)

print(f"\n=== EXTRACTED GENDER DISTRIBUTION ===")
print(f"Sex values: {X['sex'].unique()}")
print(f"Sex distribution: {X['sex'].value_counts().to_dict()}")

# Convert target to binary (1 for good credit, 0 for bad credit)
y_binary = (y['class'] == 1).astype(int)
print(f"\nTarget distribution: {y_binary.value_counts().to_dict()}")

# Remove rows with unknown gender
mask = X['sex'] != -1
X_clean = X[mask].copy()
y_clean = y_binary[mask].copy()

print(f"\nAfter removing unknown gender: {X_clean.shape}")
print(f"Final sex distribution: {X_clean['sex'].value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Data exploration
print("\n=== DATA TYPES ===")
print(X_clean.dtypes.value_counts())
print("\n=== MISSING VALUES ===")
print("Features:", X_clean.isnull().sum().sum())

# Prepare features for modeling (encode categorical variables)
from sklearn.preprocessing import LabelEncoder

# Get categorical columns (object type)
categorical_cols = X_clean.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_cols}")

# Create a copy for modeling
X_encoded = X_clean.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col != 'sex':  # sex is already encoded
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

# Split encoded data
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_encoded, y_clean, test_size=0.3, random_state=42)

# Train classification model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_enc, y_train_enc)

# Predict and calculate accuracy
y_pred = rf_model.predict(X_test_enc)
y_pred_proba = rf_model.predict_proba(X_test_enc)[:, 1]
accuracy = accuracy_score(y_test_enc, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_enc, y_pred))

# Doubly Robust ATE Estimation
def doubly_robust_ate(X, y, treatment_col='sex'):
    """
    Estimate Average Treatment Effect using Doubly Robust estimation.
    """
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Step 1: Estimate propensity scores P(T=1|X)
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T)
    if ps_model.classes_.shape[0] == 1:
        # Only one class present, use constant propensity
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
    
    # Step 2: Estimate outcome models μ₀(X) and μ₁(X)
    if len(X_features[T==0]) > 0 and len(X_features[T==1]) > 0:
        mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
        mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    else:
        # Fallback if one treatment group is empty
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
    
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

def doubly_robust_cate(X, y, treatment_col='sex'):
    """
    Estimate Conditional Average Treatment Effect using Doubly Robust estimation.
    """
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T)
    if ps_model.classes_.shape[0] == 1:
        # Only one class present, use constant propensity
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
    
    if len(X_features[T==0]) > 0 and len(X_features[T==1]) > 0:
        mu0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict_proba(X_features)[:, 1]
        mu1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict_proba(X_features)[:, 1]
    else:
        # Fallback if one treatment group is empty
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
    
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

dr_results = doubly_robust_ate(X_encoded, y_clean)

print(f"Treatment: Sex (male = 1, female = 0)")
print(f"Treated fraction: {dr_results['treated_fraction']:.3f}")
print(f"\nAverage Treatment Effect (Doubly Robust): {dr_results['ate_doubly_robust']:.4f}")
print(f"Average Treatment Effect (Regression only): {dr_results['ate_regression_only']:.4f}")
print(f"\nInterpretation: Male gender effect on credit approval probability is {dr_results['ate_doubly_robust']:.4f} (on average)")

# Run CATE estimation
print("\n" + "="*50)
print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
print("="*50)

cate_results = doubly_robust_cate(X_encoded, y_clean)

print(f"CATE Statistics:")
print(f"Mean CATE (DR): {np.mean(cate_results['cate_doubly_robust']):.4f}")
print(f"Mean CATE (Reg): {np.mean(cate_results['cate_regression_only']):.4f}")
print(f"Std CATE (Reg): {np.std(cate_results['cate_regression_only']):.4f}")

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('German Credit Classification Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax1.figure.colorbar(im, ax=ax1)
ax1.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=['Bad Credit', 'Good Credit'],
        yticklabels=['Bad Credit', 'Good Credit'],
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
ax2.hist(y_pred_proba[y_test_enc == 0], alpha=0.7, label='Bad Credit', bins=20, color='red')
ax2.hist(y_pred_proba[y_test_enc == 1], alpha=0.7, label='Good Credit', bins=20, color='blue')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Probability Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Importance
importance = pd.Series(rf_model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
indices = np.argsort(rf_model.feature_importances_)[::-1]
ax3.bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_[indices], color='orange')
ax3.set_xlabel('Features')
ax3.set_ylabel('Importance')
ax3.set_title('Feature Importance')
ax3.set_xticks(range(len(rf_model.feature_importances_)))
ax3.set_xticklabels([X_encoded.columns[i] for i in indices], rotation=45)
ax3.grid(True, alpha=0.3)

# Plot 4: Treatment Effect Distribution (sex)
treatment_1 = y_test_enc[X_test_enc['sex'] == 1]
treatment_0 = y_test_enc[X_test_enc['sex'] == 0]
ax4.hist(treatment_0, alpha=0.7, label='Female', bins=3, color='red', align='mid')
ax4.hist(treatment_1, alpha=0.7, label='Male', bins=3, color='blue', align='mid')
ax4.set_xlabel('Credit Category')
ax4.set_ylabel('Frequency')
ax4.set_title('Outcome Distribution by Gender')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Bad Credit', 'Good Credit'])

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
plt.savefig('german_credit_results.png', dpi=300, bbox_inches='tight')

print(f"\nTop 5 important features:")
print(importance.head().to_string())
print("\nMain results plot saved as 'german_credit_results.png'")

# Export data for counterfactual analysis
print("\n" + "="*50)
print("EXPORTING DATA FOR COUNTERFACTUAL ANALYSIS")
print("="*50)

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Use sex as treatment variable
treatment_col = 'sex'
train_t = X_train_enc[treatment_col].values.reshape(-1, 1)
test_t = X_test_enc[treatment_col].values.reshape(-1, 1)

# Features without treatment
train_x = X_train_enc.drop(columns=[treatment_col]).values
test_x = X_test_enc.drop(columns=[treatment_col]).values

# Outcomes (binary)
train_y = y_train_enc.values.reshape(-1, 1)
test_y = y_test_enc.values.reshape(-1, 1)

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
print(f"Treatment variable: {treatment_col} (male = 1, female = 0)")
print(f"Outcome variable: credit approval (binary classification)")
print(f"Features: {train_x.shape[1]} variables (excluding treatment)")

print("\n" + "="*50)
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Dataset: German Credit (n={len(X_encoded)})")
print(f"Treatment: Sex (male = 1, female = 0)")
print(f"Outcome: Credit approval (binary: good/bad credit)")
print(f"Model Performance: Accuracy = {accuracy:.4f}")
print(f"Average Treatment Effect: {dr_results['ate_doubly_robust']:.4f}")
print(f"Treatment prevalence: {dr_results['treated_fraction']:.1%}")
print("\nFiles generated:")
print("- german_credit_results.png (main analysis)")
print("- data/ folder with CSV files for counterfactual analysis")