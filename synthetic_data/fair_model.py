import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, combination_type="average", dataset_name="", alpha_range=None):
    """
    Calculate MSE for different linear combinations of factual and counterfactual data.
    
    Args:
        train_x: Training features
        train_y_factual: Factual training outcomes
        train_y_counterfactual: Counterfactual training outcomes
        test_x: Test features
        test_y: Test outcomes
        train_t: Training treatments
        test_t: Test treatments
        combination_type: "average", "direct", or "weighted" combination method
        dataset_name: Name of the counterfactual dataset
        alpha_range: Range of alpha values for linear combination (default: 0 to 1 with 0.1 step)
    
    Returns:
        dict: Dictionary with alpha values as keys and MSE as values
    """
    if alpha_range is None:
        alpha_range = np.arange(0, 1.1, 0.1)
    
    mse_results = {}
    
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
    
    # Pre-compute predictions for different combinations
    predictions_cache = {}
    
    for a_f in alpha_range:
        # Multiply by probability constants
        train_y_combined = a_f * (train_y_factual.iloc[:, 0]) + (1 - a_f) * ((train_y_counterfactual.iloc[:, 0] * coun_const+train_y_factual.iloc[:, 0] * fact_const))
        
        # Create a key for caching based on the actual training data
        cache_key = hash(train_y_combined.values.tobytes())
        
        if cache_key not in predictions_cache:
            print(f"  Training new model for alpha {a_f:.1f}")
            # Train Random Forest model
            rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
            rf.fit(train_x, train_y_combined.values.ravel())
            
            # Predict on both train and test sets
            train_pred = rf.predict(train_x)
            test_pred = rf.predict(test_x)
            
            predictions_cache[cache_key] = (train_pred, test_pred)
        else:
            print(f"  Using cached predictions for alpha {a_f:.1f}")
            train_pred, test_pred = predictions_cache[cache_key]
        
        # Create directory for this specific case
        case_dir = f"./predictions/{dataset_name.lower().replace(' ', '_')}_{combination_type}_alpha_{a_f:.1f}"
        os.makedirs(case_dir, exist_ok=True)
        
        # Save all required datasets for GANITE (with predictions as y values)
        train_x.to_csv(f"{case_dir}/train_x_1.0.0_continuous.csv", index=False)
        train_t.to_csv(f"{case_dir}/train_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(train_pred, columns=['0']).to_csv(f"{case_dir}/train_y_1.0.0_continuous.csv", index=False)
        test_x.to_csv(f"{case_dir}/test_x_1.0.0_continuous.csv", index=False)
        test_t.to_csv(f"{case_dir}/test_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(test_pred, columns=['0']).to_csv(f"{case_dir}/test_y_1.0.0_continuous.csv", index=False)
        
        # Save predictions comparison for both train and test sets
        # Combine train and test data for more comprehensive metrics
        combined_actual = np.concatenate([train_y_combined.values.ravel(), test_y.values.ravel()])
        combined_predicted = np.concatenate([train_pred, test_pred])
        predictions_df = pd.DataFrame({
            'actual': combined_actual,
            'predicted': combined_predicted
        })
        predictions_df.to_csv(f"{case_dir}/predictions.csv", index=False)
        
        print(f"  Saved {len(predictions_df)} predictions ({len(train_pred)} train + {len(test_pred)} test)")
        
        # Calculate MSE
        mse = mean_squared_error(test_y, test_pred)
        mse_results[a_f] = mse
        
        print(f"Alpha: {a_f:.1f}, MSE: {mse:.4f}")
    
    return mse_results

def load_ganite_counterfactuals(alpha_ganite):
    """Load GANITE counterfactuals for a specific alpha"""
    base_path_ganite = "./counterfactuals/"
    test_y_hat = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/test_y_hat_1.0.0_continous_ganite_alpha{alpha_ganite}.csv")
    train_t = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/train_t_1.0.0_continous_ganite_alpha{alpha_ganite}.csv")
    
    # Get train portion from test_y_hat (first part corresponds to training data)
    train_size = len(train_t)
    train_y_hat = test_y_hat.iloc[:train_size]
    
    # Extract counterfactuals
    train_cf_ganite = []
    for index, row in train_y_hat.iterrows():
        cf_treatment = 1 - int(train_t.iloc[index, 0])
        train_cf_ganite.append(row[str(cf_treatment)])
    
    return pd.DataFrame(train_cf_ganite, columns=['0'])

def load_cate_counterfactuals():
    """Load CATE counterfactuals"""
    try:
        # Load the potential outcomes and extract counterfactuals
        train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_continuous_cate.csv")
        train_t = pd.read_csv("./data/train_t_1.0.0_continuous.csv")
        
        # Extract counterfactuals based on treatment assignment
        train_cf_cate = []
        for i in range(len(train_potential)):
            if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
                train_cf_cate.append(train_potential.iloc[i, 1])
            else:  # Treatment group, counterfactual is Y0
                train_cf_cate.append(train_potential.iloc[i, 0])
        
        return pd.DataFrame(train_cf_cate, columns=['0'])
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return None

def run_fair_model_analysis():
    """Run fair model analysis with all 3 combination types and all counterfactual datasets"""
    # Load synthetic data
    version = "1.0.0"
    base_path = "./data/"
    
    # Load required datasets
    train_x = pd.read_csv(f"{base_path}train_x_{version}_continuous.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_continuous.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_continuous.csv")
    train_cf_manual = pd.read_csv(f"{base_path}train_cf_manual_{version}_continuous.csv")
    train_cf_random = pd.read_csv(f"{base_path}train_cf_random_{version}_continuous.csv")
    train_cf_y = pd.read_csv(f"{base_path}train_cf_y_{version}_continuous.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_continuous.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_continuous.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_continuous.csv")
    
    # Define combination types (only average)
    combination_types = ["average"]
    counterfactual_datasets = {
        "Manual": train_cf_manual,
        "Random": train_cf_random,
        "True": train_cf_y
    }
    
    # Load GANITE datasets
    ganite_alphas = [5, 6, 7, 8]
    ganite_datasets = {}
    for ganite_alpha in ganite_alphas:
        try:
            ganite_datasets[f"GANITE_α{ganite_alpha}"] = load_ganite_counterfactuals(ganite_alpha)
        except Exception as e:
            print(f"Error loading GANITE alpha {ganite_alpha}: {e}")
    
    # Load CATE counterfactuals
    cate_datasets = {}
    cate_cf = load_cate_counterfactuals()
    if cate_cf is not None:
        cate_datasets["CATE"] = cate_cf
    
    # Combine all datasets
    all_datasets = {**counterfactual_datasets, **ganite_datasets, **cate_datasets}
    
    # Create plots for each combination type
    for combo_type in combination_types:
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h']
        
        print(f"\n{'='*50}")
        print(f"COMBINATION TYPE: {combo_type.upper()}")
        print(f"{'='*50}")
        
        all_results = {}
        
        for i, (dataset_name, cf_data) in enumerate(all_datasets.items()):
            print(f"\n=== {dataset_name} Counterfactuals ({combo_type}) ===")
            mse_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, combination_type=combo_type, dataset_name=dataset_name)
            all_results[dataset_name] = mse_results
            
            alphas = list(mse_results.keys())
            plt.plot(alphas, list(mse_results.values()), 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                    linestyle='-', label=dataset_name, linewidth=2, markersize=8)
        
        plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title(f'Fair Model Performance - {combo_type.capitalize()} Combination', fontsize=14)
        plt.ylim(0, 1000)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'fair_model_{combo_type}_combination.png', dpi=300, bbox_inches='tight')
        
        # Save results for this combination type
        results_data = {'alpha': alphas}
        for dataset_name, mse_results in all_results.items():
            results_data[f'mse_{dataset_name.lower().replace(" ", "_")}'] = list(mse_results.values())
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f'fair_model_{combo_type}_results.csv', index=False)
        
        # Find optimal results
        print(f"\n=== OPTIMAL RESULTS ({combo_type.upper()}) ===")
        for dataset_name, mse_results in all_results.items():
            optimal_alpha = min(mse_results, key=mse_results.get)
            print(f"{dataset_name} - Alpha: {optimal_alpha:.1f}, MSE: {mse_results[optimal_alpha]:.4f}")

if __name__ == "__main__":
    run_fair_model_analysis()