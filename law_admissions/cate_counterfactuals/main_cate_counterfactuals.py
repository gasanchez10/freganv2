import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from numpy import corrcoef
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def doubly_robust_cate(X, y, treatment_col='T'):
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T).predict_proba(X_features)[:, 1]
    mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0]).predict(X_features)
    mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1]).predict(X_features)
    
    # Debug: Check for issues in components
    print(f"T stats: min={T.min()}, max={T.max()}, unique={np.unique(T)}")
    print(f"ps stats: min={ps.min():.6f}, max={ps.max():.6f}, has_zero={np.any(ps == 0)}, has_nan={np.isnan(ps).any()}")
    print(f"mu0 stats: min={mu0.min():.4f}, max={mu0.max():.4f}, has_nan={np.isnan(mu0).any()}")
    print(f"mu1 stats: min={mu1.min():.4f}, max={mu1.max():.4f}, has_nan={np.isnan(mu1).any()}")
    print(f"y stats: min={y.min():.4f}, max={y.max():.4f}, has_nan={np.isnan(y).any()}")
    
    # Doubly robust CATE calculation
    cate_dr = ((((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))))
    
    print(f"cate_dr stats: min={np.nanmin(cate_dr):.4f}, max={np.nanmax(cate_dr):.4f}, has_nan={np.isnan(cate_dr).any()}, nan_count={np.isnan(cate_dr).sum()}")
    print(f"cate_dr shape: {cate_dr.shape if hasattr(cate_dr, 'shape') else 'no shape'}")
    print(f"cate_dr type: {type(cate_dr)}")
    
    # Check for division by zero issues
    zero_ps = np.sum(ps == 0)
    one_ps = np.sum(ps == 1)
    print(f"Propensity score issues: zero_ps={zero_ps}, one_ps={one_ps}")
    
    if zero_ps > 0 or one_ps > 0:
        print("WARNING: Propensity scores at boundaries causing division issues")
        # Clip propensity scores to avoid division by zero
        ps_clipped = np.clip(ps, 0.01, 0.99)
        cate_dr = ((((T*(y - mu1)/ps_clipped + mu1) - ((1-T)*(y - mu0)/(1-ps_clipped) + mu0))))
        print(f"After clipping - cate_dr has_nan={np.isnan(cate_dr).any()}, nan_count={np.isnan(cate_dr).sum()}")
    
    return {
        'cate_regression': cate_dr,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

def load_law_data():
    base_path = "../data/"
    version = "1.0.0"
    
    names = ["train_x", "train_t", "train_y", "test_x", "test_t", "test_y"]
    
    data = {}
    for name in names:
        path = f"{base_path}{name}_{version}_continuous.csv"
        data[name] = pd.read_csv(path)
    
    return data

def main():
    version = "1.0.0"
    base_path = "./"
    
    data = load_law_data()
    
    train_x_df = data['train_x'].copy()
    feature_names = [f'X{i}' for i in range(train_x_df.shape[1])]
    train_x_df.columns = feature_names
    train_x_df['T'] = data['train_t'].iloc[:, 0].values
    train_y = data['train_y'].iloc[:, 0].values
    
    test_x_df = data['test_x'].copy()
    test_x_df.columns = feature_names
    test_x_df['T'] = data['test_t'].iloc[:, 0].values
    test_y = data['test_y'].iloc[:, 0].values
    
    full_X = pd.concat([train_x_df, test_x_df], ignore_index=True)
    full_y = np.concatenate([train_y, test_y])
    
    cate_results = doubly_robust_cate(full_X, full_y)
    cate_estimates = cate_results['cate_regression']
    
    train_cate = cate_estimates[:len(train_y)]
    test_cate = cate_estimates[len(train_y):]
    
    # Convert to numpy arrays for proper indexing
    if hasattr(test_cate, 'values'):
        test_cate = test_cate.values
    if hasattr(train_cate, 'values'):
        train_cate = train_cate.values
    
    train_cf_cate = []
    test_cf_cate = []
    
    for i in range(len(train_y)):
        if data['train_t'].iloc[i, 0] == 1:
            train_cf_cate.append(train_y[i] - train_cate[i])
        else:
            train_cf_cate.append(train_y[i] + train_cate[i])
    
    for i in range(len(test_y)):
        if data['test_t'].iloc[i, 0] == 1:
            test_cf_cate.append(test_y[i] - test_cate[i])
        else:
            test_cf_cate.append(test_y[i] + test_cate[i])
    
    train_cf_cate = np.array(train_cf_cate)
    test_cf_cate = np.array(test_cf_cate)
    
    train_potential_y = []
    test_potential_y = []
    
    for i in range(len(train_y)):
        if data['train_t'].iloc[i, 0] == 0:
            train_potential_y.append([train_y[i], train_cf_cate[i]])
        else:
            train_potential_y.append([train_cf_cate[i], train_y[i]])
    
    for i in range(len(test_y)):
        if data['test_t'].iloc[i, 0] == 0:
            test_potential_y.append([test_y[i], test_cf_cate[i]])
        else:
            test_potential_y.append([test_cf_cate[i], test_y[i]])
    
    train_potential_y = np.array(train_potential_y)
    test_potential_y = np.array(test_potential_y)
    
    all_potential_y = np.vstack([train_potential_y, test_potential_y])
    
    cate_folder = os.path.join(base_path, "cate_results")
    os.makedirs(cate_folder, exist_ok=True)
    
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
    
    factual_outcomes = train_y
    correlation = corrcoef(factual_outcomes, train_cf_cate)[0][1]
    
    plt.figure(figsize=(8, 6))
    plt.xlabel('Factual Outcomes')
    plt.ylabel('Counterfactual Outcomes')
    plt.title(f'CATE - Correlation: {correlation:.3f}')
    plt.scatter(factual_outcomes, train_cf_cate, alpha=0.6)
    plt.grid(True, alpha=0.3)
    save_path_graph = os.path.join(cate_folder, "correlation_cate.png")
    plt.savefig(save_path_graph, dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(cate_folder, "cate_correlation_results.txt"), "w") as f:
        f.write("CATE Counterfactuals Results:\n")
        f.write(f"Correlation Factual vs Counterfactual: {correlation:.4f}\n")
        f.write(f"Mean CATE: {np.mean(cate_estimates):.4f}\n")
        f.write(f"Std CATE: {np.std(cate_estimates):.4f}\n")
    
    print(f"CATE counterfactuals generated successfully!")
    print(f"Correlation factual vs counterfactual: {correlation:.4f}")
    print(f"Results saved to: {cate_folder}")
    
    return all_potential_y

if __name__ == '__main__':
    main()