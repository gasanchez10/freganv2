import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, combination_type="average", dataset_name="", alpha_range=None):
    if alpha_range is None:
        alpha_range = np.arange(0, 1.1, 0.1)
    
    mse_results = {}
    
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
    
    predictions_cache = {}
    
    for a_f in alpha_range:
        train_y_combined = a_f * (train_y_factual.iloc[:, 0]) + (1 - a_f) * ((train_y_counterfactual.iloc[:, 0] * coun_const+train_y_factual.iloc[:, 0] * fact_const))
        
        cache_key = hash(train_y_combined.values.tobytes())
        
        if cache_key not in predictions_cache:
            print(f"  Training new model for alpha {a_f:.1f}")
            rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
            rf.fit(train_x, train_y_combined.values.ravel())
            
            train_pred = rf.predict(train_x)
            test_pred = rf.predict(test_x)
            
            predictions_cache[cache_key] = (train_pred, test_pred)
        else:
            print(f"  Using cached predictions for alpha {a_f:.1f}")
            train_pred, test_pred = predictions_cache[cache_key]
        
        case_dir = f"./predictions/{dataset_name.lower().replace(' ', '_')}_{combination_type}_alpha_{a_f:.1f}"
        os.makedirs(case_dir, exist_ok=True)
        
        train_x.to_csv(f"{case_dir}/train_x_1.0.0_continuous.csv", index=False)
        train_t.to_csv(f"{case_dir}/train_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(train_pred, columns=['0']).to_csv(f"{case_dir}/train_y_1.0.0_continuous.csv", index=False)
        test_x.to_csv(f"{case_dir}/test_x_1.0.0_continuous.csv", index=False)
        test_t.to_csv(f"{case_dir}/test_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(test_pred, columns=['0']).to_csv(f"{case_dir}/test_y_1.0.0_continuous.csv", index=False)
        
        combined_actual = np.concatenate([train_y_combined.values.ravel(), test_y.values.ravel()])
        combined_predicted = np.concatenate([train_pred, test_pred])
        predictions_df = pd.DataFrame({
            'actual': combined_actual,
            'predicted': combined_predicted
        })
        predictions_df.to_csv(f"{case_dir}/predictions.csv", index=False)
        
        print(f"  Saved {len(predictions_df)} predictions ({len(train_pred)} train + {len(test_pred)} test)")
        
        mse = mean_squared_error(test_y, test_pred)
        mse_results[a_f] = mse
        
        print(f"Alpha: {a_f:.1f}, MSE: {mse:.4f}")
    
    return mse_results

def load_cate_counterfactuals():
    try:
        train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_continuous_cate.csv")
        train_t = pd.read_csv("./data/train_t_1.0.0_continuous.csv")
        
        train_cf_cate = []
        for i in range(len(train_potential)):
            if train_t.iloc[i, 0] == 0:
                train_cf_cate.append(train_potential.iloc[i, 1])
            else:
                train_cf_cate.append(train_potential.iloc[i, 0])
        
        return pd.DataFrame(train_cf_cate, columns=['0'])
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return None

def run_fair_model_analysis():
    version = "1.0.0"
    base_path = "./data/"
    
    train_x = pd.read_csv(f"{base_path}train_x_{version}_continuous.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_continuous.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_continuous.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_continuous.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_continuous.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_continuous.csv")
    
    combination_types = ["average"]
    
    cate_datasets = {}
    cate_cf = load_cate_counterfactuals()
    if cate_cf is not None:
        cate_datasets["CATE"] = cate_cf
    
    all_datasets = cate_datasets
    
    for combo_type in combination_types:
        plt.figure(figsize=(15, 10))
        
        colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown']
        markers = ['*', 'o', 's', '^', 'd', 'v', '<']
        
        print(f"\n{'='*50}")
        print(f"COMBINATION TYPE: {combo_type.upper()}")
        print(f"{'='*50}")
        
        all_results = {}
        alphas = None
        
        for i, (dataset_name, cf_data) in enumerate(all_datasets.items()):
            print(f"\n=== {dataset_name} Counterfactuals ({combo_type}) ===")
            mse_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, combination_type=combo_type, dataset_name=dataset_name)
            all_results[dataset_name] = mse_results
            
            if alphas is None:
                alphas = list(mse_results.keys())
            
            plt.plot(alphas, list(mse_results.values()), 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                    linestyle='-', label=dataset_name, linewidth=2, markersize=8)
        
        if alphas is not None and all_results:
            plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
            plt.ylabel('Mean Squared Error', fontsize=12)
            plt.title(f'Fair Model Performance - {combo_type.capitalize()} Combination', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'fair_model_{combo_type}_combination.png', dpi=300, bbox_inches='tight')
            
            results_data = {'alpha': alphas}
            for dataset_name, mse_results in all_results.items():
                results_data[f'mse_{dataset_name.lower().replace(" ", "_")}'] = list(mse_results.values())
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(f'fair_model_{combo_type}_results.csv', index=False)
        
        print(f"\n=== OPTIMAL RESULTS ({combo_type.upper()}) ===")
        for dataset_name, mse_results in all_results.items():
            optimal_alpha = min(mse_results, key=mse_results.get)
            print(f"{dataset_name} - Alpha: {optimal_alpha:.1f}, MSE: {mse_results[optimal_alpha]:.4f}")

if __name__ == "__main__":
    run_fair_model_analysis()