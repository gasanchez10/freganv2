import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from numpy import corrcoef
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def doubly_robust_cate(X, y, treatment_col='T'):
    """Calculate CATE using doubly robust estimation."""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores and outcome models
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Individual CATE estimates using doubly robust method
    cate_dr = ((((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))))
    cate_reg = mu1 - mu0
    
    return {
        'cate_regression': cate_dr,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

def load_synthetic_data():
    """Load synthetic data from CSV files."""
    base_path = "../data/"
    version = "1.0.0"
    
    names = ["train_x", "train_t", "train_y", "train_cf_manual", 
             "test_x", "test_t", "test_y", "test_cf_manual"]
    
    data = {}
    for name in names:
        path = f"{base_path}{name}_{version}_continuous.csv"
        data[name] = pd.read_csv(path)
    
    return data

def main():
    """Generate CATE-based counterfactuals with same structure as GANITE."""
    version = "1.0.0"
    base_path = "./"
    
    # Load data
    data = load_synthetic_data()
    
    # Prepare features for CATE estimation
    train_x_df = data['train_x'].copy()
    train_x_df.columns = ['Z1', 'Z2', 'J', 'P', 'D']
    train_x_df['T'] = data['train_t'].iloc[:, 0].values
    train_y = data['train_y'].iloc[:, 0].values
    
    test_x_df = data['test_x'].copy()
    test_x_df.columns = ['Z1', 'Z2', 'J', 'P', 'D']
    test_x_df['T'] = data['test_t'].iloc[:, 0].values
    test_y = data['test_y'].iloc[:, 0].values
    
    # Combine for full CATE estimation
    full_X = pd.concat([train_x_df, test_x_df], ignore_index=True)
    full_y = np.concatenate([train_y, test_y])
    
    # Calculate CATE
    cate_results = doubly_robust_cate(full_X, full_y)
    cate_estimates = cate_results['cate_regression']
    
    # Use regression-based CATE if doubly robust has NaN values
    if np.isnan(cate_estimates).any():
        cate_estimates = cate_results['mu1'] - cate_results['mu0']
    
    # Split CATE back to train/test
    train_cate = cate_estimates[:len(train_y)]
    test_cate = cate_estimates[len(train_y):]
    
    # Generate counterfactuals using CATE
    train_cf_cate = []
    test_cf_cate = []
    
    # For training data
    for i in range(len(train_y)):
        if data['train_t']['0'].iloc[i] == 1:  # If treated, subtract CATE
            train_cf_cate.append(train_y[i] - train_cate[i])
        else:  # If control, add CATE
            train_cf_cate.append(train_y[i] + train_cate[i])
    
    # For test data
    for i in range(len(test_y)):
        if data['test_t']['0'].iloc[i] == 1:  # If treated, subtract CATE
            test_cf_cate.append(test_y[i] - test_cate[i])
        else:  # If control, add CATE
            test_cf_cate.append(test_y[i] + test_cate[i])
    
    train_cf_cate = np.array(train_cf_cate)
    test_cf_cate = np.array(test_cf_cate)
    
    # Create potential outcomes structure (same as GANITE)
    train_potential_y = []
    test_potential_y = []
    
    for i in range(len(train_y)):
        if data['train_t']['0'].iloc[i] == 0:  # Control
            train_potential_y.append([train_y[i], train_cf_cate[i]])  # [Y0, Y1]
        else:  # Treatment
            train_potential_y.append([train_cf_cate[i], train_y[i]])  # [Y0, Y1]
    
    for i in range(len(test_y)):
        if data['test_t']['0'].iloc[i] == 0:  # Control
            test_potential_y.append([test_y[i], test_cf_cate[i]])  # [Y0, Y1]
        else:  # Treatment
            test_potential_y.append([test_cf_cate[i], test_y[i]])  # [Y0, Y1]
    
    train_potential_y = np.array(train_potential_y)
    test_potential_y = np.array(test_potential_y)
    
    # Create y_hat structure (counterfactual predictions for all)
    all_potential_y = np.vstack([train_potential_y, test_potential_y])
    
    # Save results in same structure as GANITE
    cate_folder = os.path.join(base_path, "cate_results")
    os.makedirs(cate_folder, exist_ok=True)
    
    # Save all data arrays
    data_arrays = [
        data['train_x'], data['train_t'], data['train_y'], train_potential_y,
        data['test_x'], data['test_t'], data['test_y'], test_potential_y, 
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
    
    # Calculate correlation with manual counterfactuals
    train_cf_manual = data['train_cf_manual']['0'].values[:len(train_cf_cate)]
    
    # Debug: Check for issues
    print(f"Manual CF stats: mean={np.mean(train_cf_manual):.4f}, std={np.std(train_cf_manual):.4f}, has_nan={np.isnan(train_cf_manual).any()}")
    print(f"CATE CF stats: mean={np.mean(train_cf_cate):.4f}, std={np.std(train_cf_cate):.4f}, has_nan={np.isnan(train_cf_cate).any()}")
    
    # Remove NaN values for correlation
    valid_mask = ~(np.isnan(train_cf_manual) | np.isnan(train_cf_cate))
    if valid_mask.sum() > 1 and np.std(train_cf_manual[valid_mask]) > 0 and np.std(train_cf_cate[valid_mask]) > 0:
        correlation = corrcoef(train_cf_manual[valid_mask], train_cf_cate[valid_mask])[0][1]
    else:
        correlation = np.nan
    
    # Generate correlation plot
    plt.figure(figsize=(8, 6))
    plt.xlabel('CF Manual')
    plt.ylabel('CF CATE')
    plt.title(f'CATE Counterfactuals - Correlation: {correlation:.3f}' if not np.isnan(correlation) else 'CATE Counterfactuals - Correlation: NaN')
    plt.scatter(train_cf_manual, train_cf_cate, alpha=0.6)
    plt.grid(True, alpha=0.3)
    save_path_graph = os.path.join(cate_folder, "correlation_cate.png")
    plt.savefig(save_path_graph, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation results
    with open(os.path.join(cate_folder, "cate_correlation_results.txt"), "w") as f:
        f.write("CATE Counterfactuals Results:\n")
        f.write(f"Correlation with Manual CF: {correlation:.4f}\n")
        f.write(f"Mean CATE: {np.mean(cate_estimates):.4f}\n")
        f.write(f"Std CATE: {np.std(cate_estimates):.4f}\n")
    
    print(f"CATE counterfactuals generated successfully!")
    print(f"Correlation with manual counterfactuals: {correlation:.4f}")
    print(f"Results saved to: {cate_folder}")
    
    return all_potential_y

if __name__ == '__main__':
    main()