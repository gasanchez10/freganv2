import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import corrcoef

from ganite import GANITE

def process_ganite_scenarios():
    """Process GANITE scenarios and generate counterfactuals"""
    
    predictions_dir = "../predictions"
    output_dir = "./"
    
    # Get all case directories
    case_dirs = [d for d in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, d))]
    
    # GANITE parameters
    ganite_alphas = [5]
    
    for case_dir in case_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {case_dir}")
        print(f"{'='*60}")
        
        case_path = os.path.join(predictions_dir, case_dir)
        output_case_path = os.path.join(output_dir, case_dir)
        os.makedirs(output_case_path, exist_ok=True)
        
        # Check if required files exist
        required_files = [
            "train_x_1.0.0_binary2025.csv",
            "train_t_1.0.0_binary2025.csv", 
            "train_y_1.0.0_binary2025.csv",
            "test_x_1.0.0_binary2025.csv",
            "test_t_1.0.0_binary2025.csv",
            "test_y_1.0.0_binary2025.csv"
        ]
        
        if not all(os.path.exists(os.path.join(case_path, f)) for f in required_files):
            print(f"Missing required files in {case_dir}, skipping...")
            continue
        
        try:
            # Load data for this case
            train_x = pd.read_csv(os.path.join(case_path, "train_x_1.0.0_binary2025.csv"))
            train_t = pd.read_csv(os.path.join(case_path, "train_t_1.0.0_binary2025.csv"))
            train_y = pd.read_csv(os.path.join(case_path, "train_y_1.0.0_binary2025.csv"))
            test_x = pd.read_csv(os.path.join(case_path, "test_x_1.0.0_binary2025.csv"))
            test_t = pd.read_csv(os.path.join(case_path, "test_t_1.0.0_binary2025.csv"))
            test_y = pd.read_csv(os.path.join(case_path, "test_y_1.0.0_binary2025.csv"))
            
            # Process each GANITE alpha
            for alpha in ganite_alphas:
                print(f"\nProcessing GANITE Alpha {alpha} for {case_dir}")
                
                # Create results folder
                results_folder = os.path.join(output_case_path, f"ganite_alpha_{alpha}")
                os.makedirs(results_folder, exist_ok=True)
                
                # Prepare data for GANITE
                combined_x = pd.concat([train_x, test_x], ignore_index=True)
                combined_t = pd.concat([train_t, test_t], ignore_index=True)
                combined_y = pd.concat([train_y, test_y], ignore_index=True)
                
                # Initialize and train GANITE
                ganite = GANITE(combined_x.values, combined_t.values.ravel(), combined_y.values.ravel(), alpha=alpha)
                ganite.train(iterations=10000)
                
                # Generate counterfactuals
                potential_outcomes = ganite.generate_counterfactuals()
                
                # Split back to train/test
                train_size = len(train_t)
                train_potential = potential_outcomes[:train_size]
                test_potential = potential_outcomes[train_size:]
                
                # Save all data with GANITE suffix
                version = "1.0.0"
                
                # Save training data
                train_x.to_csv(os.path.join(results_folder, f"train_x_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                train_t.to_csv(os.path.join(results_folder, f"train_t_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                train_y.to_csv(os.path.join(results_folder, f"train_y_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                
                # Save test data
                test_x.to_csv(os.path.join(results_folder, f"test_x_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                test_t.to_csv(os.path.join(results_folder, f"test_t_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                test_y.to_csv(os.path.join(results_folder, f"test_y_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                
                # Save potential outcomes
                train_potential_df = pd.DataFrame(train_potential, columns=['0', '1'])
                test_potential_df = pd.DataFrame(test_potential, columns=['0', '1'])
                
                train_potential_df.to_csv(os.path.join(results_folder, f"train_potential_y_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                test_potential_df.to_csv(os.path.join(results_folder, f"test_potential_y_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                
                # Create test_y_hat (combined train and test potential outcomes)
                combined_potential_y = pd.concat([train_potential_df, test_potential_df], ignore_index=True)
                combined_potential_y.to_csv(os.path.join(results_folder, f"test_y_hat_{version}_binary_ganite_alpha{alpha}.csv"), index=False)
                
                # Generate correlation plot
                train_pred = train_y.iloc[:, 0].values
                train_counterfactuals = []
                
                for i in range(len(train_t)):
                    if train_t.iloc[i, 0] == 1:  # If treated, counterfactual is Y0
                        train_counterfactuals.append(train_potential[i, 0])
                    else:  # If control, counterfactual is Y1
                        train_counterfactuals.append(train_potential[i, 1])
                
                train_counterfactuals = np.array(train_counterfactuals)
                correlation = corrcoef(train_pred, train_counterfactuals)[0][1]
                
                plt.figure(figsize=(8, 6))
                plt.xlabel('Train Predictions')
                plt.ylabel('GANITE Counterfactuals')
                plt.title(f'GANITE Alpha {alpha} - Correlation: {correlation:.3f}')
                plt.scatter(train_pred, train_counterfactuals, alpha=0.6)
                plt.grid(True, alpha=0.3)
                plot_path = os.path.join(results_folder, f"correlation_alpha_{alpha}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save correlation results
                with open(os.path.join(results_folder, f"ganite_alpha_{alpha}_correlation_results.txt"), "w") as f:
                    f.write(f"GANITE Alpha {alpha} Correlation Results for {case_dir}\n")
                    f.write(f"Correlation: {correlation:.4f}\n")
                
                print(f"Completed GANITE Alpha {alpha} for {case_dir} (correlation: {correlation:.3f})")
            
            print(f"Successfully processed all GANITE alphas for {case_dir}")
                
        except Exception as e:
            print(f"Error processing {case_dir}: {e}")
            continue

if __name__ == '__main__':
    process_ganite_scenarios()