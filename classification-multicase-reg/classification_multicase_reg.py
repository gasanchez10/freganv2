import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit", "boston_housing_bin"

def load_dataset(dataset_name):
    """Load dataset based on configuration"""
    if dataset_name == "synthetic_class_data":
        return load_synthetic_class_data()
    elif dataset_name == "german_credit":
        return load_german_credit()
    elif dataset_name == "boston_housing_bin":
        return load_boston_housing_bin()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_synthetic_class_data():
    """Load synthetic classification data from saved files"""
    version = "1.0.0"
    data_path = "../synthetic_class_data/data/"
    
    # Load train and test data (using binary2025 naming convention)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_binary2025.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_binary2025.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_binary2025.csv").values.ravel().astype(float)
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_binary2025.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_binary2025.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_binary2025.csv").values.ravel().astype(float)
    
    # Add treatment column to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "Synthetic Classification (Regression)", "Binary Treatment"

def load_german_credit():
    """Load German credit data from saved files (confounders-only)"""
    version = "1.0.0"
    data_path = "../german_credit/data/"
    
    # Load train and test data (confounders-only)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel().astype(float)
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel().astype(float)
    
    # Add treatment column to features (already positive ATE with confounders-only)
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "German Credit (Regression)", "Gender (Male)"

def load_boston_housing_bin():
    """Load Boston housing binary data from saved files"""
    version = "1.0.0"
    data_path = "../boston_housing_bin/data/"
    
    # Load train and test data (using binary suffix)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_binary.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_binary.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_binary.csv").values.ravel().astype(float)
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_binary.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_binary.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_binary.csv").values.ravel().astype(float)
    
    # Add treatment column to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "Boston Housing Binary (Regression)", "Charles River"

def doubly_robust_ate_regression(X, y, treatment_col='T'):
    """Calculate ATE using doubly robust estimation for regression"""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores with clipping
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    
    # Estimate outcome models
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Doubly robust ATE (correct formula)
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(T)
    }

def run_analysis():
    """Run unified classification analysis pipeline using regression approach"""
    print(f"=== CLASSIFICATION AS REGRESSION ANALYSIS: {DATASET.upper()} ===")
    
    # Load dataset (train and test data)
    X_train, y_train, t_train, X_test, y_test, t_test, dataset_name, treatment_name = load_dataset(DATASET)
    
    print(f"Dataset: {dataset_name}")
    print(f"Treatment: {treatment_name}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_test.shape[1]-1}")  # -1 for treatment column
    print(f"Treatment prevalence: {np.mean(t_test):.3f}")
    print(f"Positive class prevalence: {np.mean(y_test):.3f}")
    
    # Train model on training features (excluding treatment) using regression
    X_train_features = X_train.drop('T', axis=1)
    X_test_features = X_test.drop('T', axis=1)
    
    # Use Random Forest Regressor (treating classification as regression)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train_features, y_train)
    
    # Predictions on test set (continuous values)
    y_pred_continuous = rf.predict(X_test_features)
    
    # Convert continuous predictions to binary for classification metrics
    y_pred_binary = (y_pred_continuous > 0.5).astype(int)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred_continuous)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test.astype(int), y_pred_binary)
    try:
        # Use continuous predictions for AUC
        auc = roc_auc_score(y_test.astype(int), y_pred_continuous)
    except ValueError:
        auc = 0.5  # If only one class present
    
    print(f"\\nModel Performance (Regression Approach):")
    print(f"MSE: {mse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Calculate ATE on full dataset using regression approach
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.concatenate([y_train, y_test])
    ate_results = doubly_robust_ate_regression(X_combined, y_combined)
    
    print(f"\\nCausal Analysis (Regression):")
    print(f"ATE (Doubly Robust): {ate_results['ate_doubly_robust']:.4f}")
    print(f"ATE (Regression): {ate_results['ate_regression_only']:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Continuous vs Binary Predictions
    plt.subplot(2, 3, 1)
    plt.scatter(y_pred_continuous, y_pred_binary, alpha=0.6)
    plt.xlabel('Continuous Predictions')
    plt.ylabel('Binary Predictions')
    plt.title(f'Continuous vs Binary Predictions\\nMSE: {mse:.4f}, Acc: {accuracy:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Distribution
    plt.subplot(2, 3, 2)
    plt.hist(y_pred_continuous, bins=30, alpha=0.7, density=True, label='Predictions')
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Continuous Predictions')
    plt.ylabel('Density')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    importance = pd.Series(rf.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
    importance.head(10).plot(kind='barh')
    plt.xlabel('Importance')
    plt.title('Top 10 Features')
    
    # Plot 4: Outcome by Treatment (Continuous)
    plt.subplot(2, 3, 4)
    y_control = y_test[t_test == 0]
    y_treated = y_test[t_test == 1]
    
    control_mean = np.mean(y_control) if len(y_control) > 0 else 0
    treated_mean = np.mean(y_treated) if len(y_treated) > 0 else 0
    
    plt.bar(['Control', 'Treated'], [control_mean, treated_mean], alpha=0.7)
    plt.ylabel('Mean Outcome (Continuous)')
    plt.title(f'Outcome by Treatment\\nATE: {ate_results["ate_doubly_robust"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals Analysis
    plt.subplot(2, 3, 5)
    residuals = y_test - y_pred_continuous
    plt.scatter(y_pred_continuous, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: ATE Summary
    plt.subplot(2, 3, 6)
    ate_methods = ['Doubly Robust', 'Regression Only']
    ate_values = [ate_results['ate_doubly_robust'], ate_results['ate_regression_only']]
    plt.bar(ate_methods, ate_values, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title('ATE Comparison (Regression)')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(ate_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'Classification as Regression Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DATASET}_classification_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'dataset': dataset_name,
        'treatment': treatment_name,
        'samples': len(X_test),
        'features': X_test.shape[1]-1,
        'treatment_prevalence': np.mean(t_test),
        'positive_class_prevalence': np.mean(y_test),
        'mse': mse,
        'accuracy': accuracy,
        'auc': auc,
        'ate_doubly_robust': ate_results['ate_doubly_robust'],
        'ate_regression': ate_results['ate_regression_only']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{DATASET}_classification_regression_results.csv', index=False)
    
    print(f"\\nResults saved:")
    print(f"- {DATASET}_classification_regression_analysis.png")
    print(f"- {DATASET}_classification_regression_results.csv")
    
    return results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit", "boston_housing_bin"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python classification_multicase_reg.py [dataset_name]")
        sys.exit(1)
    
    run_analysis()