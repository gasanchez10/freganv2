import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_data"  # Options: "synthetic_data", "student_performance", "crime"

# Global variables to store computed models
_global_ps = None
_global_mu0 = None
_global_mu1 = None
_global_X_features = None
_global_T = None

def compute_models_once(X, y, treatment_col='T', dataset_name="synthetic_data"):
    """Compute propensity scores and outcome models once and store globally"""
    global _global_ps, _global_mu0, _global_mu1, _global_X_features, _global_T
    
    _global_T = X[treatment_col].astype(int)
    _global_X_features = X.drop(columns=[treatment_col])
    
    # Get optimal parameters
    ps_params = get_optimal_params(dataset_name, 'classifier')
    reg_params = get_optimal_params(dataset_name, 'regressor')
    
    # Compute once and store globally
    ps_model = RandomForestClassifier(**ps_params, random_state=42)
    _global_ps = ps_model.fit(_global_X_features, _global_T).predict_proba(_global_X_features)[:, 1]
    _global_ps = np.clip(_global_ps, 0.05, 0.95)
    
    mu0_model = RandomForestRegressor(**reg_params, random_state=42)
    mu1_model = RandomForestRegressor(**reg_params, random_state=42)
    
    _global_mu0 = mu0_model.fit(_global_X_features[_global_T==0], y[_global_T==0]).predict(_global_X_features)
    _global_mu1 = mu1_model.fit(_global_X_features[_global_T==1], y[_global_T==1]).predict(_global_X_features)

def doubly_robust_ate_cate(y, use_test_portion=False):
    """Calculate ATE and CATE using pre-computed global models"""
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    if use_test_portion:
        # Use only test portion of global models (last 300 samples)
        test_start = len(_global_T) - len(y)
        ps = _global_ps[test_start:]
        mu0 = _global_mu0[test_start:]
        mu1 = _global_mu1[test_start:]
        T = _global_T[test_start:]
    else:
        ps = _global_ps
        mu0 = _global_mu0
        mu1 = _global_mu1
        T = _global_T
    
    # Individual CATE estimates using doubly robust method
    cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    
    # Calculate ATE and CATE statistics from the same estimation
    ate_dr = np.mean(cate_dr)
    cate_mean = np.mean(cate_dr**2)
    
    return {
        'ate_doubly_robust': ate_dr,
        'cate_mean': cate_mean
    }

def get_optimal_params(dataset_name, model_type):
    """Get optimal parameters with reduced regularization for better convergence"""
    if model_type == 'classifier':
        if dataset_name == "synthetic_data":
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
        elif dataset_name == "crime":
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
        else:  # student_performance
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
    else:  # regressor
        if dataset_name == "synthetic_data":
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
        elif dataset_name == "crime":
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
        else:  # student_performance
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}



def analyze_fairness_across_alphas(dataset_name):
    """Analyze fairness metrics across alpha values for all prediction types"""
    
    # Load original data
    version = "1.0.0"
    data_path = f"../{dataset_name}/data/"
    
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").iloc[:, 0].values
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    
    # Combine train and test for full analysis
    full_x = pd.concat([train_x, test_x], ignore_index=True)
    full_t = np.concatenate([train_t, test_t])
    full_y = np.concatenate([train_y, test_y])
    
    # Add treatment column (invert for student_performance to get positive ATE)
    if dataset_name == "student_performance":
        full_x['T'] = 1 - full_t  # Invert treatment
    else:
        full_x['T'] = full_t
    
    alpha_range = np.arange(0, 1.1, 0.1)
    
    results = {
        'baseline': {'ate_dr': [], 'cate_mean': []},
        'predictions': {'ate_dr': [], 'cate_mean': []},
        'pred_counterfactuals': {'ate_dr': [], 'cate_mean': []}
    }
    
    print(f"=== FAIRNESS ANALYSIS: {dataset_name.upper()} ===")
    
    # Compute global models once using full dataset
    print("\\nComputing global models for baseline...")
    
    # Use full dataset for global model computation
    compute_models_once(full_x, full_y, dataset_name=dataset_name)
    
    # Calculate baseline using PureBaseline predictions (counterfactual=factual) at α=1.0
    pred_folder = f"./{dataset_name}_predictions"
    baseline_preds_alpha1 = pd.read_csv(f"{pred_folder}/PureBaseline_predictions_alpha_1.0/test_predictions.csv")['predictions'].values
    baseline_results = doubly_robust_ate_cate(baseline_preds_alpha1, use_test_portion=True)
    
    for a_f in alpha_range:
        print(f"\\nAnalyzing Alpha {a_f:.1f}...")
        
        # 1. Baseline Analysis (constant across alphas)
        results['baseline']['ate_dr'].append(baseline_results['ate_doubly_robust'])
        results['baseline']['cate_mean'].append(baseline_results['cate_mean'])
        
        # 2. Predictions Analysis
        try:
            pred_folder = f"./{dataset_name}_predictions"
            baseline_preds = pd.read_csv(f"{pred_folder}/Baseline_predictions_alpha_{a_f:.1f}/test_predictions.csv")['predictions'].values
            cate_preds = pd.read_csv(f"{pred_folder}/CATE_predictions_alpha_{a_f:.1f}/test_predictions.csv")['predictions'].values
            
            # Use CATE predictions as outcomes - should show convergence trend
            pred_results = doubly_robust_ate_cate(cate_preds, use_test_portion=True)
            
            results['predictions']['ate_dr'].append(pred_results['ate_doubly_robust'])
            results['predictions']['cate_mean'].append(pred_results['cate_mean'])
            
        except FileNotFoundError:
            print(f"  Warning: Predictions not found for alpha {a_f:.1f}")
            results['predictions']['ate_dr'].append(np.nan)
            results['predictions']['cate_mean'].append(np.nan)
        
        # 3. Prediction Counterfactuals Analysis
        try:
            pred_cf_folder = f"./{dataset_name}_predictions_counterfactuals"
            pred_cf = pd.read_csv(f"{pred_cf_folder}/alpha_{a_f:.1f}/test_pred_cf_{version}_continuous_alpha{a_f:.1f}.csv")['pred_counterfactual'].values
            
            # Use prediction counterfactuals as outcomes - should show convergence trend
            pred_cf_results = doubly_robust_ate_cate(pred_cf, use_test_portion=True)
            
            results['pred_counterfactuals']['ate_dr'].append(pred_cf_results['ate_doubly_robust'])
            results['pred_counterfactuals']['cate_mean'].append(pred_cf_results['cate_mean'])
            
        except FileNotFoundError:
            print(f"  Warning: Prediction counterfactuals not found for alpha {a_f:.1f}")
            results['pred_counterfactuals']['ate_dr'].append(np.nan)
            results['pred_counterfactuals']['cate_mean'].append(np.nan)
    
    return results, alpha_range

def plot_fairness_analysis(results, alpha_range, dataset_name):
    """Plot ATE and CATE across alphas for all prediction types"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'baseline': 'blue', 'predictions': 'red', 'pred_counterfactuals': 'green'}
    labels = {'baseline': 'Baseline (Original)', 'predictions': 'Predictions', 'pred_counterfactuals': 'Pred Counterfactuals'}
    
    # Plot 1: ATE Doubly Robust
    ax1 = axes[0]
    for pred_type, color in colors.items():
        ate_values = results[pred_type]['ate_dr']
        ax1.plot(alpha_range, ate_values, 'o-', color=color, label=labels[pred_type], linewidth=2, markersize=6)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('ATE (Doubly Robust)')
    ax1.set_title('Average Treatment Effect - Doubly Robust')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CATE Mean
    ax2 = axes[1]
    for pred_type, color in colors.items():
        cate_values = results[pred_type]['cate_mean']
        ax2.plot(alpha_range, cate_values, '^-', color=color, label=labels[pred_type], linewidth=2, markersize=6)
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('CATE Mean')
    ax2.set_title('Conditional Average Treatment Effect - Mean')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fairness Analysis: {dataset_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{dataset_name}_fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_fairness_results(results, alpha_range, dataset_name):
    """Save fairness analysis results to CSV"""
    
    # Prepare data for CSV
    data = {'alpha': alpha_range}
    
    for pred_type in results.keys():
        for metric in results[pred_type].keys():
            data[f'{pred_type}_{metric}'] = results[pred_type][metric]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(f'{dataset_name}_fairness_results.csv', index=False)
    
    # Define labels
    labels = {'baseline': 'Baseline (Original)', 'predictions': 'Predictions', 'pred_counterfactuals': 'Pred Counterfactuals'}
    
    # Print summary
    print(f"\\n=== FAIRNESS SUMMARY: {dataset_name.upper()} ===")
    for pred_type in results.keys():
        print(f"\\n{labels[pred_type]}:")
        print(f"  ATE (DR) range: [{np.nanmin(results[pred_type]['ate_dr']):.4f}, {np.nanmax(results[pred_type]['ate_dr']):.4f}]")
        print(f"  CATE mean range: [{np.nanmin(results[pred_type]['cate_mean']):.4f}, {np.nanmax(results[pred_type]['cate_mean']):.4f}]")

def main():
    """Run fairness analysis across alpha values"""
    print(f"=== FAIRNESS ANALYSIS PIPELINE: {DATASET.upper()} ===")
    
    # Analyze fairness across alphas
    results, alpha_range = analyze_fairness_across_alphas(DATASET)
    
    # Plot results
    plot_fairness_analysis(results, alpha_range, DATASET)
    
    # Save results
    save_fairness_results(results, alpha_range, DATASET)
    
    print(f"\\nFairness analysis completed for {DATASET}!")
    print(f"Results saved: {DATASET}_fairness_analysis.png, {DATASET}_fairness_results.csv")

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_data", "student_performance", "crime"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python fairness_analysis.py [dataset_name]")
        sys.exit(1)
    
    main()