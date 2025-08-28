import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

_global_ps = None
_global_mu0 = None
_global_mu1 = None
_global_X_features = None
_global_T = None
model_counterfactuals=True 

def compute_models_once(X, y, treatment_col='T'):
    global _global_ps, _global_mu0, _global_mu1, _global_X_features, _global_T
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    _global_T = X[treatment_col].astype(int)
    _global_X_features = X.drop(columns=[treatment_col])
    
    _global_ps = RandomForestClassifier(n_estimators=100, random_state=42).fit(_global_X_features, _global_T).predict_proba(_global_X_features)[:, 1]
    _global_mu0 = RandomForestRegressor(n_estimators=100, random_state=42).fit(_global_X_features[_global_T==0], y[_global_T==0]).predict(_global_X_features)
    _global_mu1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(_global_X_features[_global_T==1], y[_global_T==1]).predict(_global_X_features)

def doubly_robust_ate(y):
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    ate_dr = np.mean((_global_T*(y - _global_mu1)/_global_ps + _global_mu1) - ((1-_global_T)*(y - _global_mu0)/(1-_global_ps) + _global_mu0))
    
    ate_reg = np.mean(_global_mu1 - _global_mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(_global_T)
    }

def doubly_robust_cate(y):
    global _global_ps, _global_mu0, _global_mu1, _global_T
    
    cate_dr = np.mean((((_global_T*(y - _global_mu1)/_global_ps + _global_mu1) - ((1-_global_T)*(y - _global_mu0)/(1-_global_ps) + _global_mu0))**2))
    cate_reg = _global_mu1 - _global_mu0
    
    return {
        'cate_doubly_robust': cate_dr,
        'cate_regression_only': cate_reg
    }

def calculate_ate_cate_from_predictions(case_dir, models_computed=False):
    try:
        if model_counterfactuals:
            if "cate_predictions_counterfactuals" in case_dir:
                cate_dir = f"{case_dir}/cate_results"
                if not os.path.exists(cate_dir):
                    return None, None, None, None
                    
                test_y_hat = pd.read_csv(f"{cate_dir}/test_y_hat_1.0.0_continuous_cate.csv")
                test_x = pd.read_csv(f"{cate_dir}/test_x_1.0.0_continuous_cate.csv")
                test_t = pd.read_csv(f"{cate_dir}/test_t_1.0.0_continuous_cate.csv")
                
                test_y_hat = test_y_hat.tail(len(test_x))
                
                full_y = test_y_hat.iloc[:, 1].values
                full_X = test_x.copy()
                full_T = test_t.copy()
            else:
                alpha_dir = f"{case_dir}/alpha_6"
                if not os.path.exists(alpha_dir):
                    return None, None, None, None
                    
                test_y_hat = pd.read_csv(f"{alpha_dir}/test_y_hat_1.0.0_continuous_ganite_alpha6.csv")
                test_x = pd.read_csv(f"{alpha_dir}/test_x_1.0.0_continuous_ganite_alpha6.csv")
                test_t = pd.read_csv(f"{alpha_dir}/test_t_1.0.0_continuous_ganite_alpha6.csv")
                
                test_y_hat = test_y_hat.tail(len(test_x))
                
                full_y = test_y_hat.iloc[:, 1].values
                full_X = test_x.copy()
                full_T = test_t.copy()
        else:
            pred_file = f"{case_dir}/predictions.csv"
            if not os.path.exists(pred_file):
                return None, None, None, None
            
            pred_data = pd.read_csv(pred_file)
            
            test_x = pd.read_csv(f"{case_dir}/test_x_1.0.0_continuous.csv")
            test_t = pd.read_csv(f"{case_dir}/test_t_1.0.0_continuous.csv")
            
            test_pred_data = pred_data[pred_data.get('set', 'test') == 'test'] if 'set' in pred_data.columns else pred_data.tail(len(test_x))
            
            full_X = test_x.copy()
            full_T = test_t.copy()
            full_y = test_pred_data['predicted'].values
        
        full_X['T'] = full_T.iloc[:, 0]
        
        if not models_computed:
            compute_models_once(full_X, full_y)
        
        ate_results = doubly_robust_ate(full_y)
        
        cate_results = doubly_robust_cate(full_y)
        
        cate_estimates = cate_results['cate_doubly_robust']
        
        cate_treated = cate_estimates
        cate_control = cate_estimates
        cate_mean = cate_results['cate_doubly_robust']
        
        return ate_results['ate_doubly_robust'], cate_treated, cate_control, cate_mean
        
    except Exception as e:
        print(f"Error processing {case_dir}: {e}")
        return None, None, None, None

def analyze_fairness_results():
    os.makedirs("./fairness_results", exist_ok=True)
    
    if model_counterfactuals:
        case_dirs = glob("./predictions_counterfactuals/*_average_alpha_*")
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
        
        if '_alpha_' in case_name:
            alpha_val = float(case_name.split('_alpha_')[1])
        else:
            continue
            
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
    
    results_df = pd.DataFrame(results)
    
    results_df['dataset_type'] = results_df['case_name'].str.extract(r'^([\w_α\d]+)_average_alpha')
    
    results_df.loc[results_df['case_name'].str.contains('cate_average_alpha'), 'dataset_type'] = 'cate_predictions'
    
    def get_model_type(dataset_type):
        if 'ganite' in dataset_type.lower():
            return 'ganite'
        elif 'cate' in dataset_type.lower():
            return 'cate'
        else:
            return dataset_type.lower()
    
    results_df['ganite_model_type'] = results_df['dataset_type'].apply(get_model_type)
    
    results_df.to_csv("./fairness_results/complete_fairness_results.csv", index=False)
    
    dataset_types = results_df['dataset_type'].unique()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'navy']
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h', '*', '+', 'x', 'D', 'H']
    
    plt.figure(figsize=(15, 10))
    for i, dataset_type in enumerate(dataset_types):
        dataset_data = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        if not dataset_data.empty:
            plt.plot(dataset_data['alpha_value'], dataset_data['ate'], 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=f'{dataset_type.capitalize()}', linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Fair Model Combination Factor)', fontsize=12)
    plt.ylabel('Average Treatment Effect', fontsize=12)
    plt.title('Average Treatment Effect Analysis', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    
    plot_prefix = 'cf_' if model_counterfactuals else ''
    plt.savefig(f'./fairness_results/{plot_prefix}ate_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    for i, dataset_type in enumerate(dataset_types):
        dataset_data = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        if not dataset_data.empty:
            plt.plot(dataset_data['alpha_value'], dataset_data['cate_mean'], 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=f'{dataset_type.capitalize()}', linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Fair Model Combination Factor)', fontsize=12)
    plt.ylabel('Conditional Average Treatment Effect', fontsize=12)
    plt.title('Conditional Average Treatment Effect Analysis', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    
    plot_prefix = 'cf_' if model_counterfactuals else ''
    plt.savefig(f'./fairness_results/{plot_prefix}cate_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    for i, dataset_type in enumerate(dataset_types):
        dataset_data = results_df[results_df['dataset_type'] == dataset_type].sort_values('alpha_value')
        if not dataset_data.empty:
            plt.figure(figsize=(12, 8))
            plt.plot(dataset_data['alpha_value'], dataset_data['ate'], 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    linewidth=2, markersize=8)
            plt.xlabel('Alpha (Fair Model Combination Factor)', fontsize=12)
            plt.ylabel('Average Treatment Effect', fontsize=12)
            plt.title(f'ATE Analysis - {dataset_type.capitalize()}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.tight_layout()
            
            plot_prefix = 'cf_' if model_counterfactuals else ''
            plt.savefig(f'./fairness_results/{plot_prefix}ate_{dataset_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            plt.plot(dataset_data['alpha_value'], dataset_data['cate_mean'], 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    linewidth=2, markersize=8)
            plt.xlabel('Alpha (Fair Model Combination Factor)', fontsize=12)
            plt.ylabel('Conditional Average Treatment Effect', fontsize=12)
            plt.title(f'CATE Analysis - {dataset_type.capitalize()}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.tight_layout()
            
            plot_prefix = 'cf_' if model_counterfactuals else ''
            plt.savefig(f'./fairness_results/{plot_prefix}cate_{dataset_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Fairness analysis complete. Results saved in ./fairness_results/")
    print(f"Using {'counterfactual' if model_counterfactuals else 'standard'} predictions")
    print(f"Generated {len(dataset_types)} dataset type analyses")

if __name__ == "__main__":
    analyze_fairness_results()