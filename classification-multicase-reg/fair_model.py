import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

def get_optimal_rf_params(dataset_name):
    """Get optimal Random Forest parameters for dataset"""
    if dataset_name == "synthetic_class_data":
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    elif dataset_name == "german_credit":
        return {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
    elif dataset_name == "boston_housing_bin":
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    else:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }

def fair_metrics_calculator_regression(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, cf_type="", dataset_name=""):
    """Calculate MSE and accuracy using Random Forest regression with fairness regularization"""
    alpha_range = np.arange(0, 1.1, 0.1)
    results = {}
    
    # Get optimal parameters for this dataset
    params = get_optimal_rf_params(dataset_name)
    print(f"Optimizing RF parameters for {cf_type}...")
    print(f"Optimal params: {params}")
    
    for a_f in alpha_range:
        # Create combined target using probability-weighted combination (same as regression-multicase)
        combined_target = a_f * (train_y_factual.iloc[:, 0]) + (1 - a_f) * ((train_y_counterfactual.iloc[:, 0] + train_y_factual.iloc[:, 0]) / 2)
        
        # Train Random Forest regressor
        rf = RandomForestRegressor(**params)
        rf.fit(train_x.values, combined_target.values)
        
        # Predict on test set (continuous values)
        test_pred_continuous = rf.predict(test_x.values)
        
        # Convert to binary predictions for accuracy calculation
        test_pred_binary = (test_pred_continuous > 0.5).astype(int)
        
        # Calculate metrics
        mse = mean_squared_error(test_y.values.astype(float), test_pred_continuous)
        accuracy = accuracy_score(test_y.values.astype(int), test_pred_binary)
        
        # Save predictions for this alpha in dataset folder
        dataset_folder = f"./{dataset_name}_predictions"
        pred_folder = f"{dataset_folder}/{cf_type}_predictions_alpha_{a_f:.1f}"
        os.makedirs(pred_folder, exist_ok=True)
        
        # Save both continuous and binary predictions
        pd.DataFrame(test_pred_continuous, columns=['predictions_continuous']).to_csv(f"{pred_folder}/test_predictions_continuous.csv", index=False)
        pd.DataFrame(test_pred_binary, columns=['predictions']).to_csv(f"{pred_folder}/test_predictions.csv", index=False)
        
        # Also save training predictions for fairness analysis
        train_pred_continuous = rf.predict(train_x.values)
        train_pred_binary = (train_pred_continuous > 0.5).astype(int)
        pd.DataFrame(train_pred_continuous, columns=['predictions_continuous']).to_csv(f"{pred_folder}/train_predictions_continuous.csv", index=False)
        pd.DataFrame(train_pred_binary, columns=['predictions']).to_csv(f"{pred_folder}/train_predictions.csv", index=False)
        
        results[a_f] = {'mse': mse, 'accuracy': accuracy}
        
        print(f"Alpha: {a_f:.1f}, MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    
    return results

def load_cate_counterfactuals_regression(dataset):
    """Load CATE counterfactuals for specified dataset from regression results"""
    if dataset == "synthetic_class_data":
        file_suffix = "binary_reg"
    elif dataset == "boston_housing_bin":
        file_suffix = "binary_reg"
    else:  # german_credit
        file_suffix = "continuous_reg"
    
    train_potential = pd.read_csv(f"./{dataset}_cate_results/train_potential_y_1.0.0_{file_suffix}.csv")
    train_t = pd.read_csv(f"./{dataset}_cate_results/train_t_1.0.0_{file_suffix}.csv")
    
    # Extract counterfactuals based on treatment assignment
    train_cf_cate = []
    for i in range(len(train_potential)):
        if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
            train_cf_cate.append(train_potential.iloc[i, 1])
        else:  # Treatment group, counterfactual is Y0
            train_cf_cate.append(train_potential.iloc[i, 0])
    
    return pd.DataFrame(train_cf_cate, columns=['0'])

def run_fair_model_analysis_regression(dataset):
    """Run fair model analysis using regression approach for classification datasets"""
    
    # Load original data
    if dataset == "synthetic_class_data":
        version = "1.0.0"
        orig_path = f"../{dataset}/data/"
        
        train_x = pd.read_csv(f"{orig_path}train_x_{version}_binary2025.csv")
        train_y = pd.read_csv(f"{orig_path}train_y_{version}_binary2025.csv")
        train_t = pd.read_csv(f"{orig_path}train_t_{version}_binary2025.csv")
        test_x = pd.read_csv(f"{orig_path}test_x_{version}_binary2025.csv")
        test_y = pd.read_csv(f"{orig_path}test_y_{version}_binary2025.csv")
        test_t = pd.read_csv(f"{orig_path}test_t_{version}_binary2025.csv")
        
        # Load CATE counterfactuals
        train_cf_cate = load_cate_counterfactuals_regression(dataset)
        
    elif dataset == "german_credit":
        version = "1.0.0"
        orig_path = f"../{dataset}/data/"
        
        train_x = pd.read_csv(f"{orig_path}train_x_{version}_continuous.csv")
        train_y = pd.read_csv(f"{orig_path}train_y_{version}_continuous.csv")
        train_t = pd.read_csv(f"{orig_path}train_t_{version}_continuous.csv")
        test_x = pd.read_csv(f"{orig_path}test_x_{version}_continuous.csv")
        test_y = pd.read_csv(f"{orig_path}test_y_{version}_continuous.csv")
        test_t = pd.read_csv(f"{orig_path}test_t_{version}_continuous.csv")
        
        # Load CATE counterfactuals
        train_cf_cate = load_cate_counterfactuals_regression(dataset)
        
    elif dataset == "boston_housing_bin":
        version = "1.0.0"
        orig_path = f"../{dataset}/data/"
        
        train_x = pd.read_csv(f"{orig_path}train_x_{version}_binary.csv")
        train_y = pd.read_csv(f"{orig_path}train_y_{version}_binary.csv")
        train_t = pd.read_csv(f"{orig_path}train_t_{version}_binary.csv")
        test_x = pd.read_csv(f"{orig_path}test_x_{version}_binary.csv")
        test_y = pd.read_csv(f"{orig_path}test_y_{version}_binary.csv")
        test_t = pd.read_csv(f"{orig_path}test_t_{version}_binary.csv")
        
        # Load CATE counterfactuals
        train_cf_cate = load_cate_counterfactuals_regression(dataset)
    
    # Create baseline counterfactuals (counterfactual = factual)
    train_cf_baseline = train_y.copy()
    
    # Create pure baseline where counterfactual=factual for fairness analysis
    train_cf_pure_baseline = train_y.copy()
    
    datasets = {
        "Baseline": train_cf_baseline,
        "CATE": train_cf_cate,
        "PureBaseline": train_cf_pure_baseline
    }
    
    plt.figure(figsize=(15, 10))
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    print(f"=== {dataset.upper().replace('_', ' ')} FAIR MODEL REGRESSION ANALYSIS ===")
    
    all_results = {}
    
    # Plot 1: MSE
    plt.subplot(2, 2, 1)
    for i, (cf_type, cf_data) in enumerate(datasets.items()):
        print(f"\\n=== {cf_type} Counterfactuals ===")
        results = fair_metrics_calculator_regression(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, cf_type, dataset)
        all_results[cf_type] = results
        
        alphas = list(results.keys())
        mse_values = [results[alpha]['mse'] for alpha in alphas]
        plt.plot(alphas, mse_values, 
                color=colors[i], marker=markers[i], 
                linestyle='-', label=cf_type, linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Mean Squared Error vs Alpha', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    plt.subplot(2, 2, 2)
    for i, (cf_type, cf_data) in enumerate(datasets.items()):
        results = all_results[cf_type]
        alphas = list(results.keys())
        accuracy_values = [results[alpha]['accuracy'] for alpha in alphas]
        plt.plot(alphas, accuracy_values, 
                color=colors[i], marker=markers[i], 
                linestyle='-', label=cf_type, linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Alpha', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: MSE vs Accuracy Trade-off
    plt.subplot(2, 2, 3)
    for i, (cf_type, cf_data) in enumerate(datasets.items()):
        results = all_results[cf_type]
        mse_values = [results[alpha]['mse'] for alpha in results.keys()]
        accuracy_values = [results[alpha]['accuracy'] for alpha in results.keys()]
        plt.scatter(mse_values, accuracy_values, 
                   color=colors[i], marker=markers[i], 
                   label=cf_type, s=100, alpha=0.7)
    
    plt.xlabel('MSE', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('MSE vs Accuracy Trade-off', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Alpha progression
    plt.subplot(2, 2, 4)
    alphas = list(all_results['Baseline'].keys())
    alpha_labels = [f'{alpha:.1f}' for alpha in alphas]
    
    baseline_mse = [all_results['Baseline'][alpha]['mse'] for alpha in alphas]
    cate_mse = [all_results['CATE'][alpha]['mse'] for alpha in alphas]
    
    x = np.arange(len(alphas))
    width = 0.35
    
    plt.bar(x - width/2, baseline_mse, width, label='Baseline MSE', alpha=0.7, color='blue')
    plt.bar(x + width/2, cate_mse, width, label='CATE MSE', alpha=0.7, color='red')
    
    plt.xlabel('Alpha Values', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('MSE Comparison by Alpha', fontsize=14)
    plt.xticks(x, alpha_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fair Model Regression Analysis: {dataset.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{dataset}_fair_model_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_data = {'alpha': list(all_results['Baseline'].keys())}
    for cf_type, results in all_results.items():
        results_data[f'mse_{cf_type.lower()}'] = [results[alpha]['mse'] for alpha in results.keys()]
        results_data[f'accuracy_{cf_type.lower()}'] = [results[alpha]['accuracy'] for alpha in results.keys()]
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{dataset}_fair_model_regression_results.csv', index=False)
    
    # Find optimal results
    print("\\n=== OPTIMAL RESULTS ===")
    for cf_type, results in all_results.items():
        optimal_alpha_mse = min(results, key=lambda x: results[x]['mse'])
        optimal_alpha_acc = max(results, key=lambda x: results[x]['accuracy'])
        print(f"{cf_type} - Best MSE: Alpha {optimal_alpha_mse:.1f}, MSE: {results[optimal_alpha_mse]['mse']:.4f}")
        print(f"{cf_type} - Best Accuracy: Alpha {optimal_alpha_acc:.1f}, Accuracy: {results[optimal_alpha_acc]['accuracy']:.4f}")

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit", "boston_housing_bin"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python fair_model.py [dataset_name]")
        sys.exit(1)
    
    run_fair_model_analysis_regression(DATASET)