import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from numpy import corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

def get_optimal_params(dataset_name, model_type, n_samples, n_features):
    """Get optimal parameters with better regularization for classification"""
    if model_type == 'classifier':
        if dataset_name == "synthetic_class_data":
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}
        elif dataset_name == "german_credit":
            return {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5}
        else:
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}
    else:  # For outcome models in classification, we still use classifiers
        return get_optimal_params(dataset_name, 'classifier', n_samples, n_features)

def doubly_robust_cate_classification(X, y, treatment_col='T', dataset_name="synthetic_class_data"):
    """Calculate CATE using doubly robust estimation for binary classification outcomes"""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Get optimal parameters
    ps_params = get_optimal_params(dataset_name, 'classifier', len(X), X_features.shape[1])
    outcome_params = get_optimal_params(dataset_name, 'classifier', len(X), X_features.shape[1])
    
    print(f"Using parameters for {dataset_name}:")
    print(f"Propensity model: {ps_params}")
    print(f"Outcome model: {outcome_params}")
    
    # Estimate propensity scores with stronger clipping
    ps_model = RandomForestClassifier(**ps_params, random_state=42)
    ps_model.fit(X_features, T)
    
    if len(ps_model.classes_) == 1:
        # Only one treatment class, use constant propensity
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)  # Stronger clipping to avoid extreme weights
    
    # Check sample sizes
    n_control = (T==0).sum()
    n_treated = (T==1).sum()
    print(f"Sample sizes - Control: {n_control}, Treated: {n_treated}")
    
    if n_control < 10 or n_treated < 10:
        print("Warning: Very small treatment groups, using simple method")
        # Fallback to simple method
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
        cate_reg = mu1 - mu0
        return {
            'cate_doubly_robust': cate_reg,
            'cate_regression': cate_reg,
            'mu0': mu0,
            'mu1': mu1,
            'propensity_scores': ps
        }
    
    # Estimate outcome models (probabilities for classification)
    mu0_model = RandomForestClassifier(**outcome_params, random_state=42)
    mu1_model = RandomForestClassifier(**outcome_params, random_state=42)
    
    mu0_model.fit(X_features[T==0], y[T==0])
    mu1_model.fit(X_features[T==1], y[T==1])
    
    # Get probabilities for positive class
    if len(mu0_model.classes_) == 1:
        mu0 = np.full(len(X_features), mu0_model.classes_[0])
    else:
        mu0 = mu0_model.predict_proba(X_features)[:, 1]
        
    if len(mu1_model.classes_) == 1:
        mu1 = np.full(len(X_features), mu1_model.classes_[0])
    else:
        mu1 = mu1_model.predict_proba(X_features)[:, 1]
    
    # Individual CATE estimates using doubly robust method (for classification)
    cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    cate_reg = mu1 - mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression': cate_reg,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

def load_dataset(dataset_name):
    """Load dataset based on configuration"""
    if dataset_name == "synthetic_class_data":
        return load_synthetic_class_data()
    elif dataset_name == "german_credit":
        return load_german_credit()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_synthetic_class_data():
    """Load synthetic classification data from saved files"""
    version = "1.0.0"
    data_path = "../synthetic_class_data/data/"
    
    # Load train and test data (using binary2025 naming convention)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_binary2025.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_binary2025.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_binary2025.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_binary2025.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_binary2025.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_binary2025.csv").iloc[:, 0].values
    
    # Load manual counterfactuals for comparison
    train_cf_manual = pd.read_csv(f"{data_path}train_cf_manual_{version}_binary2025.csv").iloc[:, 0].values
    test_cf_manual = pd.read_csv(f"{data_path}test_cf_manual_{version}_binary2025.csv").iloc[:, 0].values
    
    return {
        'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'train_cf_manual': train_cf_manual,
        'test_x': test_x, 'test_t': test_t, 'test_y': test_y, 'test_cf_manual': test_cf_manual,
        'dataset_name': "Synthetic Classification", 'treatment_name': "Binary Treatment"
    }

def load_german_credit():
    """Load German credit data from saved files (confounders-only)"""
    version = "1.0.0"
    data_path = "../german_credit/data/"
    
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
        'dataset_name': "German Credit (Confounders)", 'treatment_name': "Gender (Male)"
    }

def generate_cate_counterfactuals_classification(data):
    """Generate CATE-based counterfactuals for classification"""
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
        dataset_name = "synthetic_class_data"
    elif "german" in dataset_name_lower:
        dataset_name = "german_credit"
    else:
        dataset_name = "synthetic_class_data"
    
    cate_results = doubly_robust_cate_classification(full_X, full_y, dataset_name=dataset_name)
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
    
    # Generate counterfactuals using CATE (as probabilities for classification)
    train_cf_cate = []
    test_cf_cate = []
    
    # For training data
    for i in range(len(train_y)):
        if train_t[i] == 1:  # If treated, subtract CATE
            cf_prob = train_y[i] - train_cate[i]
        else:  # If control, add CATE
            cf_prob = train_y[i] + train_cate[i]
        # Clip to probability range [0,1] but keep as probabilities
        train_cf_cate.append(np.clip(cf_prob, 0, 1))
    
    # For test data
    for i in range(len(test_y)):
        if test_t[i] == 1:  # If treated, subtract CATE
            cf_prob = test_y[i] - test_cate[i]
        else:  # If control, add CATE
            cf_prob = test_y[i] + test_cate[i]
        # Clip to probability range [0,1] but keep as probabilities
        test_cf_cate.append(np.clip(cf_prob, 0, 1))
    
    train_cf_cate = np.array(train_cf_cate)
    test_cf_cate = np.array(test_cf_cate)
    
    # Create potential outcomes structure (same as regression case)
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
    """Save CATE counterfactuals results for classification"""
    version = "1.0.0"
    base_path = "./"
    
    # Create results folder
    cate_folder = os.path.join(base_path, f"{DATASET}_cate_results")
    os.makedirs(cate_folder, exist_ok=True)
    
    # Prepare data for saving
    all_potential_y = np.vstack([cf_results['train_potential_y'], cf_results['test_potential_y']])
    
    # Save all data arrays (same structure as regression case)
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
    
    # Use appropriate file suffix
    file_suffix = "binary_cate" if dataset_name == "Synthetic Classification" else "continuous_cate"
    
    for name, array in zip(data_names, data_arrays):
        df = pd.DataFrame(array)
        save_path = os.path.join(cate_folder, f"{name}_{version}_{file_suffix}.csv")
        df.to_csv(save_path, index=False)
    
    # Calculate correlation between factual and counterfactual data
    train_factual = data['train_y']
    train_counterfactual = cf_results['train_cf_cate']
    
    # Keep counterfactuals as continuous probabilities (no rounding)
    train_cf_continuous = train_counterfactual
    
    # Remove NaN values for correlation
    valid_mask = ~(np.isnan(train_factual) | np.isnan(train_cf_continuous))
    correlation = np.nan
    if valid_mask.sum() > 1 and np.std(train_factual[valid_mask]) > 0 and np.std(train_cf_continuous[valid_mask]) > 0:
        correlation = corrcoef(train_factual[valid_mask], train_cf_continuous[valid_mask])[0][1]
    
    # Compare with manual counterfactuals if available
    correlation_with_manual = np.nan
    if data['train_cf_manual'] is not None:
        manual_cf = data['train_cf_manual']
        if len(manual_cf) == len(train_cf_continuous):
            correlation_with_manual = corrcoef(train_cf_continuous, manual_cf)[0][1]
    
    # Generate correlation plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Factual vs Counterfactual
    plt.subplot(1, 2, 1)
    plt.scatter(train_factual, train_cf_continuous, alpha=0.6)
    plt.xlabel('Factual Outcomes')
    plt.ylabel('Counterfactual Outcomes (Binary)')
    plt.title(f'{dataset_name} - Factual vs Counterfactual\\nCorrelation: {correlation:.3f}' if not np.isnan(correlation) else f'{dataset_name} - Factual vs Counterfactual\\nCorrelation: NaN')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Manual vs CATE counterfactuals (if available)
    plt.subplot(1, 2, 2)
    if data['train_cf_manual'] is not None:
        plt.scatter(data['train_cf_manual'], train_cf_continuous, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('Manual Counterfactuals')
        plt.ylabel('CATE Counterfactuals')
        plt.title(f'Manual vs CATE Counterfactuals\\nCorrelation: {correlation_with_manual:.3f}')
    else:
        plt.hist(cf_results['cate_estimates'], bins=30, alpha=0.7)
        plt.xlabel('CATE Estimates')
        plt.ylabel('Frequency')
        plt.title(f'CATE Distribution\\nMean: {np.mean(cf_results["cate_estimates"]):.3f}')
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
        if not np.isnan(correlation_with_manual):
            f.write(f"Correlation with Manual CF: {correlation_with_manual:.4f}\\n")
        f.write(f"Mean CATE: {np.mean(cf_results['cate_estimates']):.4f}\\n")
        f.write(f"Std CATE: {np.std(cf_results['cate_estimates']):.4f}\\n")
        f.write(f"Treatment prevalence: {np.mean(data['train_t']):.3f}\\n")
    
    return correlation, cate_folder

def main():
    """Generate CATE-based counterfactuals for classification datasets"""
    print(f"=== CATE COUNTERFACTUALS ANALYSIS: {DATASET.upper()} ===")
    
    # Load dataset
    data = load_dataset(DATASET)
    
    print(f"Dataset: {data['dataset_name']}")
    print(f"Treatment: {data['treatment_name']}")
    print(f"Train samples: {len(data['train_x'])}, Test samples: {len(data['test_x'])}")
    print(f"Treatment prevalence: {np.mean(data['train_t']):.3f}")
    
    # Generate CATE counterfactuals
    cf_results = generate_cate_counterfactuals_classification(data)
    
    # Save results
    correlation, results_folder = save_results(data, cf_results, data['dataset_name'])
    
    print(f"\\nCausal Analysis:")
    print(f"Mean CATE: {np.mean(cf_results['cate_estimates']):.4f}")
    print(f"Std CATE: {np.std(cf_results['cate_estimates']):.4f}")
    
    if not np.isnan(correlation):
        print(f"Correlation factual vs counterfactual: {correlation:.4f}")
    else:
        print("Correlation calculation failed")
    
    # Compare with manual counterfactuals if available
    if data['train_cf_manual'] is not None:
        train_cf_continuous = cf_results['train_cf_cate']
        correlation_manual = corrcoef(train_cf_continuous, data['train_cf_manual'])[0][1]
        print(f"Correlation with manual counterfactuals: {correlation_manual:.4f}")
    
    print(f"\\nResults saved to: {results_folder}")
    print(f"- CATE counterfactuals generated successfully!")
    
    return cf_results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python main_cate_counterfactuals.py [dataset_name]")
        sys.exit(1)
    
    main()