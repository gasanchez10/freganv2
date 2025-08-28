import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from numpy import corrcoef
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_data"  # Options: "synthetic_data", "student_performance", "crime", "law_admissions"

def get_optimal_params(dataset_name, model_type, n_samples, n_features):
    """Get optimal parameters with better regularization"""
    if model_type == 'classifier':
        if dataset_name == "synthetic_data":
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}
        elif dataset_name == "crime":
            return {'n_estimators': 150, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5}
        elif dataset_name == "law_admissions":
            return {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5}
        else:  # student_performance - more regularization
            return {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10}
    else:  # regressor
        if dataset_name == "synthetic_data":
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}
        elif dataset_name == "crime":
            return {'n_estimators': 150, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5}
        elif dataset_name == "law_admissions":
            return {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5}
        else:  # student_performance - more regularization
            return {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 10}

def doubly_robust_cate(X, y, treatment_col='T', dataset_name="synthetic_data"):
    """Calculate CATE using doubly robust estimation with better regularization."""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Get optimal parameters with more regularization
    ps_params = get_optimal_params(dataset_name, 'classifier', len(X), X_features.shape[1])
    reg_params = get_optimal_params(dataset_name, 'regressor', len(X), X_features.shape[1])
    
    print(f"Using regularized parameters for {dataset_name}:")
    print(f"Propensity model: {ps_params}")
    print(f"Outcome model: {reg_params}")
    
    # Estimate propensity scores with stronger clipping
    ps_model = RandomForestClassifier(**ps_params, random_state=42)
    ps = ps_model.fit(X_features, T).predict_proba(X_features)[:, 1]
    ps = np.clip(ps, 0.05, 0.95)  # Stronger clipping to avoid extreme weights
    
    # Estimate outcome models with cross-validation for student performance
    if dataset_name == "student_performance":
        # Use simpler models to avoid overfitting
        mu0_model = RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_split=30, min_samples_leaf=15, random_state=42)
        mu1_model = RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_split=30, min_samples_leaf=15, random_state=42)
    else:
        mu0_model = RandomForestRegressor(**reg_params, random_state=42)
        mu1_model = RandomForestRegressor(**reg_params, random_state=42)
    
    # Check sample sizes
    n_control = (T==0).sum()
    n_treated = (T==1).sum()
    print(f"Sample sizes - Control: {n_control}, Treated: {n_treated}")
    
    if n_control < 10 or n_treated < 10:
        print("Warning: Very small treatment groups, using regression-only CATE")
        # Fallback to regression-only method
        mu0 = mu0_model.fit(X_features[T==0], y[T==0]).predict(X_features)
        mu1 = mu1_model.fit(X_features[T==1], y[T==1]).predict(X_features)
        cate_reg = mu1 - mu0
        return {
            'cate_doubly_robust': cate_reg,
            'cate_regression': cate_reg,
            'mu0': mu0,
            'mu1': mu1,
            'propensity_scores': ps
        }
    
    mu0 = mu0_model.fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = mu1_model.fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Individual CATE estimates using doubly robust method
    cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    cate_reg = mu1 - mu0
    
    # For student performance, add noise reduction
    if dataset_name == "student_performance":
        # Smooth extreme CATE values
        cate_dr = np.clip(cate_dr, np.percentile(cate_dr, 5), np.percentile(cate_dr, 95))
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression': cate_reg,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

def load_dataset(dataset_name):
    """Load dataset based on configuration"""
    if dataset_name == "synthetic_data":
        return load_synthetic_data()
    elif dataset_name == "student_performance":
        return load_student_performance()
    elif dataset_name == "crime":
        return load_crime()
    elif dataset_name == "law_admissions":
        return load_law_admissions()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_synthetic_data():
    """Load synthetic data from saved files"""
    version = "1.0.0"
    data_path = "../synthetic_data/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").iloc[:, 0].values
    train_cf_manual = pd.read_csv(f"{data_path}train_cf_manual_{version}_continuous.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    test_cf_manual = pd.read_csv(f"{data_path}test_cf_manual_{version}_continuous.csv").iloc[:, 0].values
    
    return {
        'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'train_cf_manual': train_cf_manual,
        'test_x': test_x, 'test_t': test_t, 'test_y': test_y, 'test_cf_manual': test_cf_manual,
        'dataset_name': "Synthetic Data", 'treatment_name': "Binary Treatment"
    }

def load_student_performance():
    """Load student performance data from saved files"""
    version = "1.0.0"
    data_path = "../student_performance/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    
    # For student performance, we don't have manual counterfactuals, so we'll use None
    return {
        'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'train_cf_manual': None,
        'test_x': test_x, 'test_t': test_t, 'test_y': test_y, 'test_cf_manual': None,
        'dataset_name': "Student Performance", 'treatment_name': "Gender (Male)"
    }

def load_crime():
    """Load crime data from saved files"""
    version = "1.0.0"
    data_path = "../crime/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    
    return {
        'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'train_cf_manual': None,
        'test_x': test_x, 'test_t': test_t, 'test_y': test_y, 'test_cf_manual': None,
        'dataset_name': "Crime", 'treatment_name': "Binary Treatment"
    }

def load_law_admissions():
    """Load law admissions data from saved files"""
    version = "1.0.0"
    data_path = "../law_admissions/data/"
    
    # Load train and test data
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    
    # Invert treatment for positive ATE (consistent with regression_multicase)
    train_t = 1 - train_t
    test_t = 1 - test_t
    
    return {
        'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'train_cf_manual': None,
        'test_x': test_x, 'test_t': test_t, 'test_y': test_y, 'test_cf_manual': None,
        'dataset_name': "Law Admissions", 'treatment_name': "Race (Non-White)"
    }

def generate_cate_counterfactuals(data):
    """Generate CATE-based counterfactuals"""
    # Prepare features for CATE estimation
    train_x_df = data['train_x'].copy()
    train_x_df['T'] = data['train_t']
    train_y = data['train_y']
    
    test_x_df = data['test_x'].copy()
    test_x_df['T'] = data['test_t']
    test_y = data['test_y']
    
    # Combine for full CATE estimation
    full_X = pd.concat([train_x_df, test_x_df], ignore_index=True)
    full_y = np.concatenate([train_y, test_y])
    
    # Calculate CATE with dataset-specific optimization
    dataset_name_lower = str(data.get('dataset_name', '')).lower()
    if "synthetic" in dataset_name_lower:
        dataset_name = "synthetic_data"
    elif "crime" in dataset_name_lower:
        dataset_name = "crime"
    elif "law" in dataset_name_lower:
        dataset_name = "law_admissions"
    else:
        dataset_name = "student_performance"
    
    cate_results = doubly_robust_cate(full_X, full_y, dataset_name=dataset_name)
    cate_estimates = cate_results['cate_doubly_robust']
    
    # Use regression-based CATE if doubly robust has NaN values
    if np.isnan(cate_estimates).any():
        print("Warning: Doubly robust CATE has NaN values, using regression-based CATE")
        cate_estimates = cate_results['cate_regression']
    
    # Ensure all arrays are numpy arrays
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_t = np.array(data['train_t'])
    test_t = np.array(data['test_t'])
    cate_estimates = np.array(cate_estimates)
    
    # Split CATE back to train/test
    train_cate = cate_estimates[:len(train_y)]
    test_cate = cate_estimates[len(train_y):]
    
    # Generate counterfactuals using CATE
    train_cf_cate = []
    test_cf_cate = []
    
    # For training data
    for i in range(len(train_y)):
        if train_t[i] == 1:  # If treated, subtract CATE
            train_cf_cate.append(train_y[i] - train_cate[i])
        else:  # If control, add CATE
            train_cf_cate.append(train_y[i] + train_cate[i])
    
    # For test data
    for i in range(len(test_y)):
        if test_t[i] == 1:  # If treated, subtract CATE
            test_cf_cate.append(test_y[i] - test_cate[i])
        else:  # If control, add CATE
            test_cf_cate.append(test_y[i] + test_cate[i])
    
    train_cf_cate = np.array(train_cf_cate)
    test_cf_cate = np.array(test_cf_cate)
    
    # Create potential outcomes structure (same as GANITE)
    train_potential_y = []
    test_potential_y = []
    
    for i in range(len(train_y)):
        if train_t[i] == 0:  # Control
            train_potential_y.append([train_y[i], train_cf_cate[i]])  # [Y0, Y1]
        else:  # Treatment
            train_potential_y.append([train_cf_cate[i], train_y[i]])  # [Y0, Y1]
    
    for i in range(len(test_y)):
        if test_t[i] == 0:  # Control
            test_potential_y.append([test_y[i], test_cf_cate[i]])  # [Y0, Y1]
        else:  # Treatment
            test_potential_y.append([test_cf_cate[i], test_y[i]])  # [Y0, Y1]
    
    train_potential_y = np.array(train_potential_y)
    test_potential_y = np.array(test_potential_y)
    
    return {
        'train_cf_cate': train_cf_cate,
        'test_cf_cate': test_cf_cate,
        'train_potential_y': train_potential_y,
        'test_potential_y': test_potential_y,
        'cate_estimates': cate_estimates,
        'cate_results': cate_results
    }

def save_results(data, cf_results, dataset_name):
    """Save CATE counterfactuals results"""
    version = "1.0.0"
    base_path = "./"
    
    # Create results folder
    cate_folder = os.path.join(base_path, f"{DATASET}_cate_results")
    os.makedirs(cate_folder, exist_ok=True)
    
    # Prepare data for saving
    all_potential_y = np.vstack([cf_results['train_potential_y'], cf_results['test_potential_y']])
    
    # Save all data arrays
    data_arrays = [
        data['train_x'], data['train_t'].reshape(-1, 1), data['train_y'].reshape(-1, 1), cf_results['train_potential_y'],
        data['test_x'], data['test_t'].reshape(-1, 1), data['test_y'].reshape(-1, 1), cf_results['test_potential_y'], 
        all_potential_y
    ]
    data_names = [
        "train_x", "train_t", "train_y", "train_potential_y",
        "test_x", "test_t", "test_y", "test_potential_y", 
        "test_y_hat"
    ]
    
    for name, array in zip(data_names, data_arrays):
        df = pd.DataFrame(array)
        save_path = os.path.join(cate_folder, f"{name}_{version}_continuous_cate.csv")
        df.to_csv(save_path, index=False)
    
    # Calculate correlation between factual and counterfactual data
    train_factual = data['train_y']
    train_counterfactual = cf_results['train_cf_cate']
    
    # Remove NaN values for correlation
    valid_mask = ~(np.isnan(train_factual) | np.isnan(train_counterfactual))
    correlation = np.nan
    if valid_mask.sum() > 1 and np.std(train_factual[valid_mask]) > 0 and np.std(train_counterfactual[valid_mask]) > 0:
        correlation = corrcoef(train_factual[valid_mask], train_counterfactual[valid_mask])[0][1]
    
    # Generate correlation plot between factual and counterfactual
    plt.figure(figsize=(8, 6))
    plt.scatter(train_factual, train_counterfactual, alpha=0.6)
    
    # Add correlation line if correlation is valid
    if not np.isnan(correlation):
        z = np.polyfit(train_factual, train_counterfactual, 1)
        p = np.poly1d(z)
        plt.plot(train_factual, p(train_factual), "r--", alpha=0.8, linewidth=2)
    
    plt.xlabel('Factual Outcomes')
    plt.ylabel('Counterfactual Outcomes')
    plt.title(f'{dataset_name} - Factual vs Counterfactual\nCorrelation: {correlation:.3f}' if not np.isnan(correlation) else f'{dataset_name} - Factual vs Counterfactual\nCorrelation: NaN')
    
    plt.grid(True, alpha=0.3)
    save_path_graph = os.path.join(cate_folder, "correlation_cate.png")
    plt.savefig(save_path_graph, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation results
    with open(os.path.join(cate_folder, "cate_correlation_results.txt"), "w") as f:
        f.write(f"CATE Counterfactuals Results - {dataset_name}:\\n")
        if not np.isnan(correlation):
            f.write(f"Correlation Factual vs Counterfactual: {correlation:.4f}\\n")
        else:
            f.write("Correlation calculation failed\\n")
        f.write(f"Mean CATE: {np.mean(cf_results['cate_estimates']):.4f}\\n")
        f.write(f"Std CATE: {np.std(cf_results['cate_estimates']):.4f}\\n")
        f.write(f"Treatment prevalence: {np.mean(data['train_t']):.3f}\\n")
    
    return correlation, cate_folder

def main():
    """Generate CATE-based counterfactuals for regression datasets"""
    print(f"=== CATE COUNTERFACTUALS ANALYSIS: {DATASET.upper()} ===")
    
    # Load dataset
    data = load_dataset(DATASET)
    
    print(f"Dataset: {data['dataset_name']}")
    print(f"Treatment: {data['treatment_name']}")
    print(f"Train samples: {len(data['train_x'])}, Test samples: {len(data['test_x'])}")
    print(f"Treatment prevalence: {np.mean(data['train_t']):.3f}")
    
    # Generate CATE counterfactuals
    cf_results = generate_cate_counterfactuals(data)
    
    # Save results
    correlation, results_folder = save_results(data, cf_results, data['dataset_name'])
    
    print(f"\\nCausal Analysis:")
    print(f"Mean CATE: {np.mean(cf_results['cate_estimates']):.4f}")
    print(f"Std CATE: {np.std(cf_results['cate_estimates']):.4f}")
    
    if not np.isnan(correlation):
        print(f"Correlation factual vs counterfactual: {correlation:.4f}")
    else:
        print("Correlation calculation failed")
    
    print(f"\\nResults saved to: {results_folder}")
    print(f"- CATE counterfactuals generated successfully!")
    
    return cf_results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_data", "student_performance", "crime", "law_admissions"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python main_cate_counterfactuals.py [dataset_name]")
        sys.exit(1)
    
    main()