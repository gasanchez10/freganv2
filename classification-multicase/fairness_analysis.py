import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

# Global variables to store computed models
_global_ps = None
_global_mu0 = None
_global_mu1 = None
_global_X_features = None
_global_T = None

def compute_models_once(X, y, treatment_col='T', dataset_name="synthetic_class_data"):
    """Compute propensity scores and outcome models once and store globally"""
    global _global_ps, _global_mu0, _global_mu1, _global_X_features, _global_T
    
    _global_T = X[treatment_col].astype(int)
    _global_X_features = X.drop(columns=[treatment_col])
    
    # Get optimal parameters for classification
    ps_params = get_optimal_params(dataset_name, 'classifier')
    outcome_params = get_optimal_params(dataset_name, 'classifier')
    
    # Compute propensity scores
    ps_model = RandomForestClassifier(**ps_params, random_state=42)
    ps_model.fit(_global_X_features, _global_T)
    
    if len(ps_model.classes_) == 1:
        _global_ps = np.full(len(_global_X_features), 0.5)
    else:
        _global_ps = ps_model.predict_proba(_global_X_features)[:, 1]
        _global_ps = np.clip(_global_ps, 0.05, 0.95)
    
    # Compute outcome models
    mu0_model = RandomForestClassifier(**outcome_params, random_state=42)
    mu1_model = RandomForestClassifier(**outcome_params, random_state=42)
    
    if len(_global_X_features[_global_T==0]) > 0 and len(_global_X_features[_global_T==1]) > 0:
        mu0_model.fit(_global_X_features[_global_T==0], y[_global_T==0])
        mu1_model.fit(_global_X_features[_global_T==1], y[_global_T==1])
        
        # Get probabilities for positive class
        if len(mu0_model.classes_) == 1:
            _global_mu0 = np.full(len(_global_X_features), mu0_model.classes_[0])
        else:
            _global_mu0 = mu0_model.predict_proba(_global_X_features)[:, 1]
            
        if len(mu1_model.classes_) == 1:
            _global_mu1 = np.full(len(_global_X_features), mu1_model.classes_[0])
        else:
            _global_mu1 = mu1_model.predict_proba(_global_X_features)[:, 1]
    else:
        _global_mu0 = np.full(len(_global_X_features), np.mean(y[_global_T==0]) if len(y[_global_T==0]) > 0 else 0.5)
        _global_mu1 = np.full(len(_global_X_features), np.mean(y[_global_T==1]) if len(y[_global_T==1]) > 0 else 0.5)

def doubly_robust_ate_cate_classification(y, use_test_portion=False):
    """Calculate ATE and CATE using pre-computed global models for classification"""
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    if use_test_portion:
        # Use only test portion of global models
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
    
    # Individual CATE estimates using doubly robust method for classification
    cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    
    # Calculate ATE and CATE statistics
    ate_dr = np.mean(cate_dr)
    cate_mean = np.mean(cate_dr**2)
    
    return {
        'ate_doubly_robust': ate_dr,
        'cate_mean': cate_mean
    }

def get_optimal_params(dataset_name, model_type):
    """Get optimal parameters for classification models"""
    if dataset_name == "synthetic_class_data":
        return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}
    elif dataset_name == "german_credit":
        return {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5}
    else:
        return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}

def analyze_fairness_across_alphas(dataset_name):
    """Analyze fairness metrics across alpha values for all prediction types"""
    
    # Load original data
    if dataset_name == "synthetic_class_data":
        data_path = f"../{dataset_name}/data/"
        train_x = pd.read_csv(f"{data_path}train_x_1.0.0_binary2025.csv")
        train_t = pd.read_csv(f"{data_path}train_t_1.0.0_binary2025.csv").iloc[:, 0].values
        train_y = pd.read_csv(f"{data_path}train_y_1.0.0_binary2025.csv").iloc[:, 0].values
        test_x = pd.read_csv(f"{data_path}test_x_1.0.0_binary2025.csv")
        test_t = pd.read_csv(f"{data_path}test_t_1.0.0_binary2025.csv").iloc[:, 0].values
        test_y = pd.read_csv(f"{data_path}test_y_1.0.0_binary2025.csv").iloc[:, 0].values
    else:  # german_credit
        data_path = f"../{dataset_name}/data/"
        train_x = pd.read_csv(f"{data_path}train_x_1.0.0_continuous.csv")
        train_t = pd.read_csv(f"{data_path}train_t_1.0.0_continuous.csv").iloc[:, 0].values
        train_y = pd.read_csv(f"{data_path}train_y_1.0.0_continuous.csv").iloc[:, 0].values
        test_x = pd.read_csv(f"{data_path}test_x_1.0.0_continuous.csv")
        test_t = pd.read_csv(f"{data_path}test_t_1.0.0_continuous.csv").iloc[:, 0].values
        test_y = pd.read_csv(f"{data_path}test_y_1.0.0_continuous.csv").iloc[:, 0].values
    
    # Combine train and test for full analysis
    full_x = pd.concat([train_x, test_x], ignore_index=True)
    full_t = np.concatenate([train_t, test_t])
    full_y = np.concatenate([train_y, test_y])
    
    # Add treatment column (no inversion needed for classification datasets)
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
    compute_models_once(full_x, full_y, dataset_name=dataset_name)
    
    # Calculate baseline using full dataset outcomes
    baseline_results = doubly_robust_ate_cate_classification(full_y, use_test_portion=False)
    
    for a_f in alpha_range:
        print(f"\\nAnalyzing Alpha {a_f:.1f}...")
        
        # 1. Baseline Analysis (constant across alphas)
        results['baseline']['ate_dr'].append(baseline_results['ate_doubly_robust'])
        results['baseline']['cate_mean'].append(baseline_results['cate_mean'])
        
        # 2. Predictions Analysis (using combined train+test)
        try:
            pred_folder = f"./{dataset_name}_predictions"
            train_preds = pd.read_csv(f"{pred_folder}/CATE_predictions_alpha_{a_f:.1f}/train_predictions.csv")['predictions'].values
            test_preds = pd.read_csv(f"{pred_folder}/CATE_predictions_alpha_{a_f:.1f}/test_predictions.csv")['predictions'].values
            combined_preds = np.concatenate([train_preds, test_preds])
            
            # Use CATE predictions as outcomes on full dataset
            pred_results = doubly_robust_ate_cate_classification(combined_preds, use_test_portion=False)
            
            results['predictions']['ate_dr'].append(pred_results['ate_doubly_robust'])
            results['predictions']['cate_mean'].append(pred_results['cate_mean'])
            
        except FileNotFoundError:
            print(f"  Warning: Predictions not found for alpha {a_f:.1f}")
            results['predictions']['ate_dr'].append(np.nan)
            results['predictions']['cate_mean'].append(np.nan)
        
        # 3. Prediction Counterfactuals Analysis (using combined train+test)
        try:
            pred_cf_folder = f"./{dataset_name}_predictions_counterfactuals"
            file_suffix = "binary2025" if dataset_name == "synthetic_class_data" else "continuous"
            train_pred_cf = pd.read_csv(f"{pred_cf_folder}/alpha_{a_f:.1f}/train_pred_cf_1.0.0_{file_suffix}_alpha{a_f:.1f}.csv")['pred_counterfactual'].values
            test_pred_cf = pd.read_csv(f"{pred_cf_folder}/alpha_{a_f:.1f}/test_pred_cf_1.0.0_{file_suffix}_alpha{a_f:.1f}.csv")['pred_counterfactual'].values
            combined_pred_cf = np.concatenate([train_pred_cf, test_pred_cf])
            
            # Use prediction counterfactuals as outcomes on full dataset
            pred_cf_results = doubly_robust_ate_cate_classification(combined_pred_cf, use_test_portion=False)
            
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
        ax2.plot(alpha_range, cate_values, 'o-', color=color, label=labels[pred_type], linewidth=2, markersize=6)
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('CATE Mean')
    ax2.set_title('Conditional Average Treatment Effect - Mean')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fairness Analysis: {dataset_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{dataset_name}_fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run fairness analysis pipeline for classification datasets"""
    print(f"=== FAIRNESS ANALYSIS PIPELINE: {DATASET.upper()} ===")
    
    # Run fairness analysis
    results, alpha_range = analyze_fairness_across_alphas(DATASET)
    
    # Plot results
    plot_fairness_analysis(results, alpha_range, DATASET)
    
    # Save detailed results
    results_data = {'alpha': alpha_range}
    for pred_type in results:
        results_data[f'{pred_type}_ate_dr'] = results[pred_type]['ate_dr']
        results_data[f'{pred_type}_cate_mean'] = results[pred_type]['cate_mean']
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{DATASET}_fairness_results.csv', index=False)
    
    # Print summary
    print(f"\\n=== FAIRNESS SUMMARY: {DATASET.upper()} ===")
    
    print(f"\\nBaseline (Original):")
    baseline_ate_range = [min(results['baseline']['ate_dr']), max(results['baseline']['ate_dr'])]
    baseline_cate_range = [min(results['baseline']['cate_mean']), max(results['baseline']['cate_mean'])]
    print(f"  ATE (DR) range: [{baseline_ate_range[0]:.4f}, {baseline_ate_range[1]:.4f}]")
    print(f"  CATE mean range: [{baseline_cate_range[0]:.4f}, {baseline_cate_range[1]:.4f}]")
    
    print(f"\\nPredictions:")
    pred_ate_clean = [x for x in results['predictions']['ate_dr'] if not np.isnan(x)]
    pred_cate_clean = [x for x in results['predictions']['cate_mean'] if not np.isnan(x)]
    if pred_ate_clean and pred_cate_clean:
        pred_ate_range = [min(pred_ate_clean), max(pred_ate_clean)]
        pred_cate_range = [min(pred_cate_clean), max(pred_cate_clean)]
        print(f"  ATE (DR) range: [{pred_ate_range[0]:.4f}, {pred_ate_range[1]:.4f}]")
        print(f"  CATE mean range: [{pred_cate_range[0]:.4f}, {pred_cate_range[1]:.4f}]")
    
    print(f"\\nPred Counterfactuals:")
    pred_cf_ate_clean = [x for x in results['pred_counterfactuals']['ate_dr'] if not np.isnan(x)]
    pred_cf_cate_clean = [x for x in results['pred_counterfactuals']['cate_mean'] if not np.isnan(x)]
    if pred_cf_ate_clean and pred_cf_cate_clean:
        pred_cf_ate_range = [min(pred_cf_ate_clean), max(pred_cf_ate_clean)]
        pred_cf_cate_range = [min(pred_cf_cate_clean), max(pred_cf_cate_clean)]
        print(f"  ATE (DR) range: [{pred_cf_ate_range[0]:.4f}, {pred_cf_ate_range[1]:.4f}]")
        print(f"  CATE mean range: [{pred_cf_cate_range[0]:.4f}, {pred_cf_cate_range[1]:.4f}]")
    
    print(f"\\nFairness analysis completed for {DATASET}!")
    print(f"Results saved: {DATASET}_fairness_analysis.png, {DATASET}_fairness_results.csv")

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python fairness_analysis.py [dataset_name]")
        sys.exit(1)
    
    main()