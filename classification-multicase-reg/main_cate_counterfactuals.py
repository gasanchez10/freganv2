import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from numpy import corrcoef
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit", "boston_housing_bin"

def doubly_robust_ate_regression(X, y, treatment_col='T'):
    """Calculate ATE using doubly robust estimation for regression"""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores using classification (treatment is binary)
    from sklearn.ensemble import RandomForestClassifier
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ps_model.fit(X_features, T)
    
    if len(ps_model.classes_) == 1:
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
    
    # Estimate outcome models
    if len(X_features[T==0]) > 0 and len(X_features[T==1]) > 0:
        mu0_model = RandomForestRegressor(n_estimators=100, random_state=42)
        mu1_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        mu0_model.fit(X_features[T==0], y[T==0])
        mu1_model.fit(X_features[T==1], y[T==1])
        
        mu0 = mu0_model.predict(X_features)
        mu1 = mu1_model.predict(X_features)
    else:
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
    
    # Doubly robust ATE
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(T)
    }

def generate_cate_counterfactuals_regression(dataset_name):
    """Generate CATE counterfactuals using regression approach for classification datasets"""
    
    print(f"=== CATE COUNTERFACTUALS REGRESSION: {dataset_name.upper()} ===")
    
    # Load data
    if dataset_name == "synthetic_class_data":
        version = "1.0.0"
        data_path = f"../{dataset_name}/data/"
        
        train_x = pd.read_csv(f"{data_path}train_x_{version}_binary2025.csv")
        train_t = pd.read_csv(f"{data_path}train_t_{version}_binary2025.csv").values.ravel()
        train_y = pd.read_csv(f"{data_path}train_y_{version}_binary2025.csv").values.ravel().astype(float)
        
        test_x = pd.read_csv(f"{data_path}test_x_{version}_binary2025.csv")
        test_t = pd.read_csv(f"{data_path}test_t_{version}_binary2025.csv").values.ravel()
        test_y = pd.read_csv(f"{data_path}test_y_{version}_binary2025.csv").values.ravel().astype(float)
        
        file_suffix = "binary_reg"
        
    elif dataset_name == "german_credit":
        version = "1.0.0"
        data_path = f"../{dataset_name}/data/"
        
        train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
        train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
        train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel().astype(float)
        
        test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
        test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
        test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel().astype(float)
        
        file_suffix = "continuous_reg"
        
    elif dataset_name == "boston_housing_bin":
        version = "1.0.0"
        data_path = f"../{dataset_name}/data/"
        
        train_x = pd.read_csv(f"{data_path}train_x_{version}_binary.csv")
        train_t = pd.read_csv(f"{data_path}train_t_{version}_binary.csv").values.ravel()
        train_y = pd.read_csv(f"{data_path}train_y_{version}_binary.csv").values.ravel().astype(float)
        
        test_x = pd.read_csv(f"{data_path}test_x_{version}_binary.csv")
        test_t = pd.read_csv(f"{data_path}test_t_{version}_binary.csv").values.ravel()
        test_y = pd.read_csv(f"{data_path}test_y_{version}_binary.csv").values.ravel().astype(float)
        
        file_suffix = "binary_reg"
    
    # Add treatment to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    print(f"Train samples: {len(train_x)}, Test samples: {len(test_x)}")
    print(f"Treatment prevalence: {np.mean(train_t):.3f}")
    
    # Calculate ATE on combined data
    X_combined = pd.concat([train_x, test_x], ignore_index=True)
    y_combined = np.concatenate([train_y, test_y])
    ate_results = doubly_robust_ate_regression(X_combined, y_combined)
    
    print(f"ATE (Doubly Robust): {ate_results['ate_doubly_robust']:.4f}")
    print(f"ATE (Regression): {ate_results['ate_regression_only']:.4f}")
    
    # Train CATE models using regression
    X_train_features = train_x.drop('T', axis=1)
    X_test_features = test_x.drop('T', axis=1)
    
    # Separate by treatment groups
    treated_mask = train_t == 1
    control_mask = train_t == 0
    
    # Train outcome models for each treatment group
    if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
        # Model for treated (T=1)
        model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_treated.fit(X_train_features[treated_mask], train_y[treated_mask])
        
        # Model for control (T=0)
        model_control = RandomForestRegressor(n_estimators=100, random_state=42)
        model_control.fit(X_train_features[control_mask], train_y[control_mask])
        
        # Generate potential outcomes for training data
        train_y0_potential = model_control.predict(X_train_features)
        train_y1_potential = model_treated.predict(X_train_features)
        
        # Generate potential outcomes for test data
        test_y0_potential = model_control.predict(X_test_features)
        test_y1_potential = model_treated.predict(X_test_features)
        
        print(f"Generated potential outcomes successfully")
        
    else:
        print("Warning: Insufficient data in treatment groups")
        return {}
    
    # Create output directory
    output_dir = f"./{dataset_name}_cate_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data and potential outcomes
    pd.DataFrame(train_x.drop('T', axis=1)).to_csv(f"{output_dir}/train_x_{version}_{file_suffix}.csv", index=False)
    pd.DataFrame(train_t, columns=['T']).to_csv(f"{output_dir}/train_t_{version}_{file_suffix}.csv", index=False)
    pd.DataFrame(train_y, columns=['y']).to_csv(f"{output_dir}/train_y_{version}_{file_suffix}.csv", index=False)
    
    # Save potential outcomes
    potential_outcomes_train = pd.DataFrame({
        'y0_potential': train_y0_potential,
        'y1_potential': train_y1_potential
    })
    potential_outcomes_train.to_csv(f"{output_dir}/train_potential_y_{version}_{file_suffix}.csv", index=False)
    
    # Save test data and potential outcomes
    pd.DataFrame(test_x.drop('T', axis=1)).to_csv(f"{output_dir}/test_x_{version}_{file_suffix}.csv", index=False)
    pd.DataFrame(test_t, columns=['T']).to_csv(f"{output_dir}/test_t_{version}_{file_suffix}.csv", index=False)
    pd.DataFrame(test_y, columns=['y']).to_csv(f"{output_dir}/test_y_{version}_{file_suffix}.csv", index=False)
    
    # Save potential outcomes
    potential_outcomes_test = pd.DataFrame({
        'y0_potential': test_y0_potential,
        'y1_potential': test_y1_potential
    })
    potential_outcomes_test.to_csv(f"{output_dir}/test_potential_y_{version}_{file_suffix}.csv", index=False)
    
    # Calculate correlation between potential outcomes
    train_correlation = corrcoef(train_y0_potential, train_y1_potential)[0][1]
    test_correlation = corrcoef(test_y0_potential, test_y1_potential)[0][1]
    
    # Create correlation plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(train_y0_potential, train_y1_potential, alpha=0.6)
    plt.xlabel('Y0 Potential (Control)')
    plt.ylabel('Y1 Potential (Treated)')
    plt.title(f'Training CATE Correlation: {train_correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_y0_potential, test_y1_potential, alpha=0.6)
    plt.xlabel('Y0 Potential (Control)')
    plt.ylabel('Y1 Potential (Treated)')
    plt.title(f'Test CATE Correlation: {test_correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'CATE Regression Analysis: {dataset_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_cate_reg.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation results
    correlation_results = {
        'train_correlation': train_correlation,
        'test_correlation': test_correlation,
        'ate_doubly_robust': ate_results['ate_doubly_robust'],
        'ate_regression': ate_results['ate_regression_only']
    }
    
    with open(f"{output_dir}/cate_correlation_results_reg.txt", 'w') as f:
        f.write(f"CATE Regression Correlation Results for {dataset_name}\n")
        f.write(f"Train correlation: {train_correlation:.4f}\n")
        f.write(f"Test correlation: {test_correlation:.4f}\n")
        f.write(f"ATE (Doubly Robust): {ate_results['ate_doubly_robust']:.4f}\n")
        f.write(f"ATE (Regression): {ate_results['ate_regression_only']:.4f}\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"Train correlation: {train_correlation:.4f}")
    print(f"Test correlation: {test_correlation:.4f}")
    
    return correlation_results

def main():
    """Generate CATE counterfactuals using regression for classification datasets"""
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit", "boston_housing_bin"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        return
    
    # Generate CATE counterfactuals
    results = generate_cate_counterfactuals_regression(DATASET)
    
    return results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit", "boston_housing_bin"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python main_cate_counterfactuals.py [dataset_name]")
        sys.exit(1)
    
    main()