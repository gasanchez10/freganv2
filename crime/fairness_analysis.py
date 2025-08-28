import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Global variables to store computed models
_global_ps = None
_global_mu0 = None
_global_mu1 = None
_global_X_features = None
_global_T = None
model_counterfactuals = False

def compute_models_once(X, y, treatment_col='T'):
    """Compute propensity scores and outcome models once"""
    global _global_ps, _global_mu0, _global_mu1, _global_X_features, _global_T
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    _global_T = X[treatment_col].astype(int)
    _global_X_features = X.drop(columns=[treatment_col])
    
    # Compute propensity scores
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(_global_X_features, _global_T)
    ps_proba = ps_model.predict_proba(_global_X_features)
    _global_ps = ps_proba[:, 1] if ps_proba.shape[1] > 1 else ps_proba[:, 0]
    
    # Compute outcome models for regression
    mu0_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(_global_X_features[_global_T==0], y[_global_T==0])
    _global_mu0 = mu0_model.predict(_global_X_features)
    
    mu1_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(_global_X_features[_global_T==1], y[_global_T==1])
    _global_mu1 = mu1_model.predict(_global_X_features)
    
    # Clip propensity scores
    _global_ps = np.clip(_global_ps, 0.01, 0.99)

def doubly_robust_ate(y):
    """Doubly Robust ATE estimation using pre-computed models"""
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    ate_dr = np.mean((_global_T*(y - _global_mu1)/_global_ps + _global_mu1) -  ((1-_global_T)*(y - _global_mu0)/(1-_global_ps) + _global_mu0))
    
    ate_reg = np.mean(_global_mu1 - _global_mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(_global_T)
    }

def doubly_robust_cate(y):
    """Doubly Robust CATE estimation using pre-computed models"""
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    cate_dr = np.mean((((_global_T*(y - _global_mu1)/_global_ps + _global_mu1) - ((1-_global_T)*(y - _global_mu0)/(1-_global_ps) + _global_mu0))**2))
    cate_reg = _global_mu1 - _global_mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression_only': cate_reg
    }

def calculate_ate_cate_from_predictions(case_dir, models_computed=False):
    """Calculate ATE and CATE using doubly robust methods from prediction data (test data only)"""
    try:
        if model_counterfactuals:
            # Check if this is a CATE predictions counterfactuals case
            if "cate_predictions_counterfactuals" in case_dir:
                # For CATE predictions counterfactuals, read from cate_results subdirectory
                cate_dir = f"{case_dir}/cate_results"
                if not os.path.exists(cate_dir):
                    return None, None, None, None
                    
                # Load test data only
                test_y_hat = pd.read_csv(f"{cate_dir}/test_y_hat_1.0.0_continuous_cate.csv")
                test_x = pd.read_csv(f"{cate_dir}/test_x_1.0.0_continuous_cate.csv")
                test_t = pd.read_csv(f"{cate_dir}/test_t_1.0.0_continuous_cate.csv")
                
                # Take only test portion
                test_y_hat = test_y_hat.tail(len(test_x))
                
                # Use test data only
                full_X = test_x.copy()
                full_T = test_t.copy()
                
                # Use counterfactual predictions for test data (continuous values)
                full_y = test_y_hat.iloc[:, 1].values  # Y1 column for counterfactuals
            else:
                # For regular predictions, use predictions.csv (test data only)
                pred_file = f"{case_dir}/predictions.csv"
                if not os.path.exists(pred_file):
                    return None, None, None, None
                
                # Load prediction data (actual vs predicted)
                pred_data = pd.read_csv(pred_file)
                
                # Load test features and treatment from the case directory
                test_x = pd.read_csv(f"{case_dir}/test_x_1.0.0_continuous.csv")
                test_t = pd.read_csv(f"{case_dir}/test_t_1.0.0_continuous.csv")
                
                # Use test data only
                full_X = test_x.copy()
                full_T = test_t.copy()
                
                # Filter predictions to test set only
                test_pred_data = pred_data.tail(len(test_x))
                
                # Check if predictions have variation
                pred_values = test_pred_data['predicted'].values
                if len(np.unique(pred_values)) == 1:
                    print(f"Warning: All predictions are {pred_values[0]}. Using original test data.")
                    # Load original test data
                    orig_test_y = pd.read_csv("./data/test_y_1.0.0_continuous.csv")
                    full_y = orig_test_y.iloc[:, 0].values
                else:
                    full_y = pred_values
        else:
            # Original prediction file logic (test data only)
            pred_file = f"{case_dir}/predictions.csv"
            if not os.path.exists(pred_file):
                return None, None, None, None
            
            # Load prediction data (actual vs predicted)
            pred_data = pd.read_csv(pred_file)
            
            # Load test features and treatment from the case directory
            test_x = pd.read_csv(f"{case_dir}/test_x_1.0.0_continuous.csv")
            test_t = pd.read_csv(f"{case_dir}/test_t_1.0.0_continuous.csv")
            
            # Use test data only
            full_X = test_x.copy()
            full_T = test_t.copy()
            
            # Filter predictions to test set only
            test_pred_data = pred_data.tail(len(test_x))
            full_y = test_pred_data['predicted'].values
        
        # Add treatment column to features
        full_X['T'] = full_T.iloc[:, 0]
        
        # Compute models only once for the first case
        if not models_computed:
            compute_models_once(full_X, full_y)
        
        # Calculate doubly robust ATE using pre-computed models
        ate_results = doubly_robust_ate(full_y)
        
        # Calculate doubly robust CATE using pre-computed models
        cate_results = doubly_robust_cate(full_y)
        
        # Calculate CATE statistics by treatment group
        cate_estimates = cate_results['cate_doubly_robust']
        
        cate_treated = cate_estimates
        cate_control = cate_estimates
        cate_mean = cate_results['cate_doubly_robust']
        
        return ate_results['ate_doubly_robust'], cate_treated, cate_control, cate_mean
        
    except Exception as e:
        print(f"Error processing {case_dir}: {e}")
        return None, None, None, None

def analyze_fairness_results():
    """Analyze fairness results from all prediction cases"""
    
    # Create fairness_results directory
    os.makedirs("./fairness_results", exist_ok=True)
    
    # Find all prediction case directories
    if model_counterfactuals:
        case_dirs = glob("./predictions/*_average_alpha_*")
        cate_case_dirs = glob("./cate_predictions_counterfactuals/*_average_alpha_*")
        case_dirs.extend(cate_case_dirs)
    else:
        case_dirs = glob("./predictions/*_average_alpha_*")
    
    results = []
    models_computed = False
    
    for case_dir in case_dirs:
        if not os.path.isdir(case_dir):
            continue
            
        case_name = os.path.basename(case_dir)
        
        # Extract alpha value from case name
        if '_alpha_' in case_name:
            alpha_val = float(case_name.split('_alpha_')[1])
        else:
            continue
            
        # Calculate ATE and CATE from predictions (doubly robust)
        ate_pred, cate_treated_pred, cate_control_pred, cate_mean_pred = calculate_ate_cate_from_predictions(case_dir, False)
        
        if ate_pred is not None:
            result_entry = {
                'case_name': case_name,
                'alpha_value': alpha_val,
                'ate': ate_pred,
                'cate_treated': cate_treated_pred,
                'cate_control': cate_control_pred,
                'cate_mean': cate_mean_pred,
                'cate_difference': abs(cate_treated_pred - cate_control_pred)
            }
            results.append(result_entry)
    
    if not results:
        print("No prediction results found!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Extract dataset type from case name
    def extract_dataset_type(case_name):
        if 'cate_average_alpha' in case_name:
            return 'cate_predictions'
        elif 'baseline_average_alpha' in case_name:
            return 'baseline'
        else:
            return case_name.split('_average_alpha')[0]
    
    results_df['dataset_type'] = results_df['case_name'].apply(extract_dataset_type)
    
    # Add model_type based on dataset_type
    def get_model_type(dataset_type):
        if pd.isna(dataset_type):
            return 'unknown'
        dataset_type = str(dataset_type)
        if 'cate' in dataset_type.lower():
            return 'cate'
        else:
            return dataset_type.lower()
    
    results_df['model_type'] = results_df['dataset_type'].apply(get_model_type)
    
    # Save complete results
    results_df.to_csv('./fairness_results/complete_fairness_results.csv', index=False)
    
    # Create visualizations
    create_fairness_visualizations(results_df)
    
    print(f"Fairness analysis completed! Processed {len(results)} cases.")
    print(f"Results saved to fairness_results/complete_fairness_results.csv")

def create_fairness_visualizations(results_df):
    """Create fairness visualization plots"""
    
    dataset_types = results_df['dataset_type'].unique()
    plot_prefix = 'cf_' if model_counterfactuals else ''
    
    # Individual plots for each dataset type
    for dataset_type in dataset_types:
        if pd.isna(dataset_type):
            continue
        dataset_type_str = str(dataset_type)
        subset = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        
        # ATE plot
        plt.figure(figsize=(10, 6))
        plt.plot(subset['alpha_value'], subset['ate'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Alpha Value')
        plt.ylabel('Average Treatment Effect (ATE)')
        plt.title(f'ATE vs Alpha - {dataset_type_str.title()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./fairness_results/{plot_prefix}ate_{dataset_type_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # CATE plot
        plt.figure(figsize=(10, 6))
        plt.plot(subset['alpha_value'], subset['cate_mean'], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Alpha Value')
        plt.ylabel('Conditional Average Treatment Effect (CATE)')
        plt.title(f'CATE vs Alpha - {dataset_type_str.title()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./fairness_results/{plot_prefix}cate_{dataset_type_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Combined plots
    plt.figure(figsize=(15, 10))
    
    # ATE combined
    plt.subplot(2, 1, 1)
    for dataset_type in dataset_types:
        if pd.isna(dataset_type):
            continue
        dataset_type_str = str(dataset_type)
        subset = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        plt.plot(subset['alpha_value'], subset['ate'], 'o-', label=dataset_type_str.title(), linewidth=2, markersize=6)
    plt.xlabel('Alpha Value')
    plt.ylabel('Average Treatment Effect (ATE)')
    plt.title('ATE vs Alpha - Crime')
    # Set y-axis limits to zoom in on the variation
    ate_min = results_df['ate'].min() * 0.95
    ate_max = results_df['ate'].max() * 1.05
    plt.ylim(ate_min, ate_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CATE combined
    plt.subplot(2, 1, 2)
    for dataset_type in dataset_types:
        if pd.isna(dataset_type):
            continue
        dataset_type_str = str(dataset_type)
        subset = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        plt.plot(subset['alpha_value'], subset['cate_mean'], 'o-', label=dataset_type_str.title(), linewidth=2, markersize=6)
    plt.xlabel('Alpha Value')
    plt.ylabel('Conditional Average Treatment Effect (CATE)')
    plt.title('CATE vs Alpha - Crime')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./fairness_results/{plot_prefix}combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_fairness_results()