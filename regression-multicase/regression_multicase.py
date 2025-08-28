import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_data"  # Options: "synthetic_data", "student_performance", "crime"

def load_dataset(dataset_name):
    """Load dataset based on configuration"""
    if dataset_name == "synthetic_data":
        return load_synthetic_data()
    elif dataset_name == "student_performance":
        return load_student_performance()
    elif dataset_name == "crime":
        return load_crime()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_synthetic_data():
    """Load synthetic data from saved files"""
    version = "1.0.0"
    data_path = "../synthetic_data/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel()
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel()
    
    # Add treatment column to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "Synthetic Data", "Binary Treatment"

def load_student_performance():
    """Load student performance data from saved files"""
    version = "1.0.0"
    data_path = "../student_performance/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel()
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel()
    
    # Add treatment column to features (invert for positive ATE)
    train_x['T'] = 1 - train_t  # Invert treatment
    test_x['T'] = 1 - test_t   # Invert treatment
    
    return train_x, train_y, 1 - train_t, test_x, test_y, 1 - test_t, "Student Performance", "Gender (Female)"

def load_crime():
    """Load crime data from saved files"""
    version = "1.0.0"
    data_path = "../crime/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel()
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel()
    
    # Add treatment column to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "Crime", "Binary Treatment"

def doubly_robust_ate(X, y, treatment_col='T'):
    """Calculate ATE using doubly robust estimation"""
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
    """Run unified analysis pipeline"""
    print(f"=== UNIFIED REGRESSION ANALYSIS: {DATASET.upper()} ===")
    
    # Load dataset (train and test data)
    X_train, y_train, t_train, X_test, y_test, t_test, dataset_name, treatment_name = load_dataset(DATASET)
    
    print(f"Dataset: {dataset_name}")
    print(f"Treatment: {treatment_name}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_test.shape[1]-1}")  # -1 for treatment column
    print(f"Treatment prevalence: {np.mean(t_test):.3f}")
    
    # Train model on training features (excluding treatment)
    X_train_features = X_train.drop('T', axis=1)
    X_test_features = X_test.drop('T', axis=1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_features, y_train)
    
    # Predictions on test set
    y_pred = rf.predict(X_test_features)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Calculate ATE on full dataset (to match original analysis)
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.concatenate([y_train, y_test])
    ate_results = doubly_robust_ate(X_combined, y_combined)
    
    print(f"\nCausal Analysis:")
    print(f"ATE (Doubly Robust): {ate_results['ate_doubly_robust']:.4f}")
    print(f"ATE (Regression): {ate_results['ate_regression_only']:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted\nMSE: {mse:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(2, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    importance = pd.Series(rf.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
    importance.head(10).plot(kind='barh')
    plt.xlabel('Importance')
    plt.title('Top 10 Features')
    
    # Plot 4: Outcome by Treatment
    plt.subplot(2, 3, 4)
    y_control = y_test[t_test == 0]
    y_treated = y_test[t_test == 1]
    plt.boxplot([y_control, y_treated], labels=['Control', 'Treated'])
    plt.ylabel('Outcome')
    plt.title(f'Outcome by Treatment\nATE: {ate_results["ate_doubly_robust"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Treatment Distribution
    plt.subplot(2, 3, 5)
    plt.hist([y_control, y_treated], bins=20, alpha=0.7, label=['Control', 'Treated'], density=True)
    plt.xlabel('Outcome')
    plt.ylabel('Density')
    plt.title('Outcome Distribution by Treatment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: ATE Summary
    plt.subplot(2, 3, 6)
    ate_methods = ['Doubly Robust', 'Regression']
    ate_values = [ate_results['ate_doubly_robust'], ate_results['ate_regression_only']]
    plt.bar(ate_methods, ate_values, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title('ATE Comparison')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(ate_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'Unified Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DATASET}_unified_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'dataset': dataset_name,
        'treatment': treatment_name,
        'samples': len(X_test),
        'features': X_test.shape[1]-1,
        'treatment_prevalence': np.mean(t_test),
        'mse': mse,
        'rmse': rmse,
        'ate_doubly_robust': ate_results['ate_doubly_robust'],
        'ate_regression': ate_results['ate_regression_only']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{DATASET}_unified_results.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"- {DATASET}_unified_analysis.png")
    print(f"- {DATASET}_unified_results.csv")
    
    return results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_data", "student_performance", "crime"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python regression_multicase.py [dataset_name]")
        sys.exit(1)
    
    run_analysis()