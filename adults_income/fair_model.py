import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, combination_type="average", dataset_name="", alpha_range=None):
    """
    Calculate MSE for different linear combinations of factual and counterfactual data.
    """
    if alpha_range is None:
        alpha_range = np.arange(0, 1.1, 0.1)
    
    mse_results = {}
    
    for a_f in alpha_range:
        # Only average combination: a_f * factual + (1 - a_f) * average(factual, counterfactual)
        train_y_combined = a_f * train_y_factual + (1 - a_f) * ((train_y_counterfactual + train_y_factual) / 2)
        
        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
        rf.fit(train_x, train_y_combined.values.ravel())
        
        # Predict on test set
        test_pred = rf.predict(test_x)
        
        # Create directory for this specific case
        case_dir = f"./predictions/{dataset_name.lower().replace(' ', '_')}_{combination_type}_alpha_{a_f:.1f}"
        os.makedirs(case_dir, exist_ok=True)
        
        # Save all required datasets for GANITE
        train_x.to_csv(f"{case_dir}/train_x_1.0.0_continuous.csv", index=False)
        train_t.to_csv(f"{case_dir}/train_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(rf.predict(train_x), columns=['0']).to_csv(f"{case_dir}/train_y_1.0.0_continuous.csv", index=False)
        test_x.to_csv(f"{case_dir}/test_x_1.0.0_continuous.csv", index=False)
        test_t.to_csv(f"{case_dir}/test_t_1.0.0_continuous.csv", index=False)
        pd.DataFrame(test_pred, columns=['0']).to_csv(f"{case_dir}/test_y_1.0.0_continuous.csv", index=False)
        
        # Calculate MSE
        mse = mean_squared_error(test_y, test_pred)
        mse_results[a_f] = mse
        
        print(f"Alpha: {a_f:.1f}, MSE: {mse:.4f}")
    
    return mse_results

def load_ganite_counterfactuals(alpha_ganite):
    """Load GANITE counterfactuals for a specific alpha"""
    base_path_ganite = "./counterfactuals/"
    test_y_hat = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/test_y_hat_1.0.0_continuous_ganite_alpha{alpha_ganite}.csv")
    train_t = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/train_t_1.0.0_continuous_ganite_alpha{alpha_ganite}.csv")
    
    # Get train portion from test_y_hat (first part corresponds to training data)
    train_size = len(train_t)
    train_y_hat = test_y_hat.iloc[:train_size]
    
    # Extract counterfactuals
    train_cf_ganite = []
    for index, row in train_y_hat.iterrows():
        cf_treatment = 1 - int(train_t.iloc[index, 0])
        train_cf_ganite.append(row[str(cf_treatment)])
    
    return pd.DataFrame(train_cf_ganite, columns=['0'])

def run_fair_model_analysis():
    """Run fair model analysis with GANITE counterfactuals only"""
    # Load adults income data
    version = "1.0.0"
    base_path = "./data/"
    
    # Load required datasets
    train_x = pd.read_csv(f"{base_path}train_x_{version}_continuous.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_continuous.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_continuous.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_continuous.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_continuous.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_continuous.csv")
    
    # Define combination types (only average)
    combination_types = ["average"]
    
    # Add real y as reference case
    ganite_datasets = {"Real_Y": train_y}
    
    # Load GANITE datasets
    ganite_alphas = [0, 1, 2, 3, 4, 5]
    for ganite_alpha in ganite_alphas:
        try:
            ganite_datasets[f"GANITE_α{ganite_alpha}"] = load_ganite_counterfactuals(ganite_alpha)
        except Exception as e:
            print(f"Error loading GANITE alpha {ganite_alpha}: {e}")
    
    # Create plots for each combination type
    for combo_type in combination_types:
        plt.figure(figsize=(15, 10))
        
        colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown']
        markers = ['*', 'o', 's', '^', 'd', 'v', '<']
        
        print(f"\n{'='*50}")
        print(f"COMBINATION TYPE: {combo_type.upper()}")
        print(f"{'='*50}")
        
        all_results = {}
        alphas = None
        
        for i, (dataset_name, cf_data) in enumerate(ganite_datasets.items()):
            print(f"\n=== {dataset_name} Counterfactuals ({combo_type}) ===")
            mse_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, combination_type=combo_type, dataset_name=dataset_name)
            all_results[dataset_name] = mse_results
            
            if alphas is None:
                alphas = list(mse_results.keys())
            
            marker_size = 12 if dataset_name == 'Real_Y' else 8
            line_width = 3 if dataset_name == 'Real_Y' else 2
            plt.plot(alphas, list(mse_results.values()), 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                    linestyle='-', label=dataset_name, linewidth=line_width, markersize=marker_size)
        
        if alphas is not None and all_results:
            plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
            plt.ylabel('Mean Squared Error', fontsize=12)
            plt.title(f'Fair Model Performance - {combo_type.capitalize()} Combination', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Use log scale to make Real_Y visible
            plt.tight_layout()
            plt.savefig(f'fair_model_{combo_type}_combination.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save results for this combination type
            results_data = {'alpha': alphas}
            for dataset_name, mse_results in all_results.items():
                results_data[f'mse_{dataset_name.lower().replace(" ", "_")}'] = list(mse_results.values())
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(f'fair_model_{combo_type}_results.csv', index=False)
        else:
            plt.close()
            print(f"No data available for {combo_type} combination")
        
        # Find optimal results
        if all_results:
            print(f"\n=== OPTIMAL RESULTS ({combo_type.upper()}) ===")
            for dataset_name, mse_results in all_results.items():
                optimal_alpha = min(mse_results, key=mse_results.get)
                print(f"{dataset_name} - Alpha: {optimal_alpha:.1f}, MSE: {mse_results[optimal_alpha]:.4f}")

if __name__ == "__main__":
    run_fair_model_analysis()