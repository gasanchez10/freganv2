import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# CONFIGURATION - Change this variable to switch datasets
DATASET = "student_performance"  # Options: "synthetic_data", "student_performance", "crime", "law_admissions"

def get_optimal_rf_params(X, y, dataset_name):
    """Get optimal Random Forest parameters for dataset"""
    if dataset_name == "synthetic_data":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    elif dataset_name == "crime":
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 8, 10],
            'min_samples_split': [5, 10, 15]
        }
    else:  # student_performance
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 10],
            'min_samples_split': [5, 10, 20]
        }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, cf_type="", dataset_name=""):
    """Calculate MSE for baseline vs CATE counterfactuals with optimized RF params"""
    alpha_range = np.arange(0, 1.1, 0.1)
    mse_results = {}
    
    # Get optimal RF parameters for this dataset
    print(f"Optimizing RF parameters for {cf_type}...")
    optimal_params = get_optimal_rf_params(train_x, train_y_factual.iloc[:, 0], dataset_name)
    print(f"Optimal params: {optimal_params}")
    
    # Calculate probability constants
    probunos = train_t.iloc[:, 0].sum() / len(train_t)
    probceros = (len(train_t) - train_t.iloc[:, 0].sum()) / len(train_t)
    
    fact_const = []
    coun_const = []
    for i in train_t.iloc[:, 0]:
        if i == 1:
            fact_const.append(probunos)
            coun_const.append(probceros)
        else:
            fact_const.append(probceros)
            coun_const.append(probunos)
    
    for a_f in alpha_range:
        # Probability-weighted combination
        train_y_combined = a_f * (train_y_factual.iloc[:, 0]) + (1 - a_f) * ((train_y_counterfactual.iloc[:, 0]  + train_y_factual.iloc[:, 0] )/2)
        
        # Train Random Forest model with optimal parameters
        rf = RandomForestRegressor(**optimal_params, random_state=42)
        rf.fit(train_x, train_y_combined.values.ravel())
        
        # Predict on test set
        test_pred = rf.predict(test_x)
        
        # Save predictions for this alpha in dataset folder
        dataset_folder = f"./{dataset_name}_predictions"
        pred_folder = f"{dataset_folder}/{cf_type}_predictions_alpha_{a_f:.1f}"
        os.makedirs(pred_folder, exist_ok=True)
        
        # Save predictions and training data
        pd.DataFrame(test_pred, columns=['predictions']).to_csv(f"{pred_folder}/test_predictions.csv", index=False)
        pd.DataFrame(train_y_combined.values, columns=['train_combined']).to_csv(f"{pred_folder}/train_combined.csv", index=False)
        
        # Calculate MSE
        mse = mean_squared_error(test_y, test_pred)
        mse_results[a_f] = mse
        
        print(f"Alpha: {a_f:.1f}, MSE: {mse:.4f}")
    
    return mse_results

def load_cate_counterfactuals(dataset):
    """Load CATE counterfactuals for specified dataset from local results"""
    train_potential = pd.read_csv(f"./{dataset}_cate_results/train_potential_y_1.0.0_continuous_cate.csv")
    train_t = pd.read_csv(f"./{dataset}_cate_results/train_t_1.0.0_continuous_cate.csv")
    
    # Extract counterfactuals based on treatment assignment
    train_cf_cate = []
    for i in range(len(train_potential)):
        if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
            train_cf_cate.append(train_potential.iloc[i, 1])
        else:  # Treatment group, counterfactual is Y0
            train_cf_cate.append(train_potential.iloc[i, 0])
    
    return pd.DataFrame(train_cf_cate, columns=['0'])

def run_fair_model_analysis(dataset):
    """Run fair model analysis comparing baseline vs CATE counterfactuals"""
    # Load original data for proper comparison
    version = "1.0.0"
    orig_path = f"../{dataset}/data/"
    
    train_x = pd.read_csv(f"{orig_path}train_x_{version}_continuous.csv")
    train_y = pd.read_csv(f"{orig_path}train_y_{version}_continuous.csv")
    train_t = pd.read_csv(f"{orig_path}train_t_{version}_continuous.csv")
    test_x = pd.read_csv(f"{orig_path}test_x_{version}_continuous.csv")
    test_y = pd.read_csv(f"{orig_path}test_y_{version}_continuous.csv")
    test_t = pd.read_csv(f"{orig_path}test_t_{version}_continuous.csv")
    
    # Create baseline counterfactuals (counterfactual = factual)
    train_cf_baseline = train_y.copy()
    
    # Create pure baseline where counterfactual=factual for fairness analysis
    train_cf_pure_baseline = train_y.copy()
    
    # Load CATE counterfactuals
    train_cf_cate = load_cate_counterfactuals(dataset)
    
    datasets = {
        "Baseline": train_cf_baseline,
        "CATE": train_cf_cate,
        "PureBaseline": train_cf_pure_baseline
    }
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    print(f"=== {dataset.upper().replace('_', ' ')} FAIR MODEL ANALYSIS ===")
    
    all_results = {}
    
    for i, (cf_type, cf_data) in enumerate(datasets.items()):
        print(f"\n=== {cf_type} Counterfactuals ===")
        mse_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, cf_type, dataset)
        all_results[cf_type] = mse_results
        
        alphas = list(mse_results.keys())
        plt.plot(alphas, list(mse_results.values()), 
                color=colors[i], marker=markers[i], 
                linestyle='-', label=cf_type, linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title(f'{dataset.replace("_", " ").title()} - Baseline vs CATE Counterfactuals', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{dataset}_fair_model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save results
    results_data = {'alpha': list(all_results['Baseline'].keys())}
    for cf_type, mse_results in all_results.items():
        results_data[f'mse_{cf_type.lower()}'] = list(mse_results.values())
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{dataset}_fair_model_results.csv', index=False)
    
    # Find optimal results
    print("\n=== OPTIMAL RESULTS ===")
    for cf_type, mse_results in all_results.items():
        optimal_alpha = min(mse_results, key=mse_results.get)
        print(f"{cf_type} - Alpha: {optimal_alpha:.1f}, MSE: {mse_results[optimal_alpha]:.4f}")

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_data", "student_performance", "crime", "law_admissions"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python fair_model.py [dataset_name]")
        sys.exit(1)
    
    run_fair_model_analysis(DATASET)