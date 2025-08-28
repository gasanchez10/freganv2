import argparse
import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from numpy import corrcoef

# Add counterfactuals directory to path
sys.path.append('../counterfactuals')
from ganite import ganite

def process_manual_scenarios():
    """Process only manual scenarios through GANITE and store counterfactuals"""
    
    predictions_dir = "../predictions"
    output_dir = "./"
    
    # Only process folders that start with 'manual'
    case_dirs = [d for d in os.listdir(predictions_dir) if d.startswith('manual') and os.path.isdir(os.path.join(predictions_dir, d))]
    
    for case_dir in case_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {case_dir}")
        print(f"{'='*60}")
        
        case_path = os.path.join(predictions_dir, case_dir)
        output_case_path = os.path.join(output_dir, case_dir)
        os.makedirs(output_case_path, exist_ok=True)
        
        # Check if required files exist
        required_files = [
            "train_x_1.0.0_continuous.csv",
            "train_t_1.0.0_continuous.csv", 
            "train_y_1.0.0_continuous.csv",
            "test_x_1.0.0_continuous.csv",
            "test_t_1.0.0_continuous.csv",
            "test_y_1.0.0_continuous.csv"
        ]
        
        if not all(os.path.exists(os.path.join(case_path, f)) for f in required_files):
            print(f"Missing required files in {case_dir}, skipping...")
            continue
        
        try:
            # Load data for this case
            train_x = pd.read_csv(os.path.join(case_path, "train_x_1.0.0_continuous.csv"))
            train_t = pd.read_csv(os.path.join(case_path, "train_t_1.0.0_continuous.csv"))
            train_y = pd.read_csv(os.path.join(case_path, "train_y_1.0.0_continuous.csv"))
            test_x = pd.read_csv(os.path.join(case_path, "test_x_1.0.0_continuous.csv"))
            test_t = pd.read_csv(os.path.join(case_path, "test_t_1.0.0_continuous.csv"))
            test_y = pd.read_csv(os.path.join(case_path, "test_y_1.0.0_continuous.csv"))
            
            # Process with different alpha values
            alphas = [6]
            
            for alpha in alphas:
                print(f"Processing alpha {alpha} for {case_dir}")
                
                # Configure GANITE parameters
                parameters = {
                    'h_dim': 30,
                    'iteration': 10000,
                    'batch_size': 256,
                    'alpha': alpha
                }
                
                # Train GANITE model
                test_y_hat = ganite(train_x, train_t, train_y, test_x, test_y, test_t, parameters)
                
                # Create alpha-specific folder
                alpha_folder = os.path.join(output_case_path, f"alpha_{alpha}")
                os.makedirs(alpha_folder, exist_ok=True)
                
                # Save results
                version = "1.0.0"
                data_arrays = [train_x, train_t, train_y, test_x, test_t, test_y, test_y_hat]
                data_names = ["train_x", "train_t", "train_y", "test_x", "test_t", "test_y", "test_y_hat"]
                
                for name, data in zip(data_names, data_arrays):
                    df = pd.DataFrame(data)
                    save_path = os.path.join(alpha_folder, f"{name}_{version}_continuous_ganite_alpha{alpha}.csv")
                    df.to_csv(save_path, index=False)
                
                # Extract and save counterfactuals
                train_size = len(train_t)
                train_y_hat_ganite = test_y_hat[:train_size]
                
                train_cf_ganite = []
                for k in range(len(train_t)):
                    cf_treatment = 1 - int(train_t.iloc[k, 0])
                    train_cf_ganite.append(train_y_hat_ganite[k][cf_treatment])
                
                # Save counterfactuals
                cf_df = pd.DataFrame(train_cf_ganite, columns=['0'])
                cf_path = os.path.join(alpha_folder, f"train_cf_ganite_{version}_continuous_alpha{alpha}.csv")
                cf_df.to_csv(cf_path, index=False)
                
                # Generate correlation plot between predictions and counterfactuals
                train_pred = train_y.iloc[:, 0].values
                if len(train_cf_ganite) > 0:
                    correlation = corrcoef(train_pred, train_cf_ganite)[0][1]
                else:
                    correlation = 0.0
                
                plt.figure(figsize=(8, 6))
                plt.xlabel('Train Predictions')
                plt.ylabel('GANITE Counterfactuals')
                plt.title(f'Alpha {alpha} - Correlation: {correlation:.3f}')
                plt.scatter(train_pred, train_cf_ganite, alpha=0.6)
                plt.grid(True, alpha=0.3)
                plot_path = os.path.join(alpha_folder, f"correlation_alpha_{alpha}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Completed alpha {alpha} for {case_dir} (correlation: {correlation:.3f})")
            
            # Save summary of optimal alphas for this case
            summary_path = os.path.join(output_case_path, "optimal_alpha_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Summary for {case_dir}\n")
                f.write("Alpha\tCorrelation\n")
                for alpha in alphas:
                    alpha_folder = os.path.join(output_case_path, f"alpha_{alpha}")
                    if os.path.exists(alpha_folder):
                        f.write(f"{alpha}\t{correlation:.4f}\n")
            
            print(f"Successfully processed {case_dir}")
                
        except Exception as e:
            print(f"Error processing {case_dir}: {e}")
            continue

if __name__ == '__main__':
    process_manual_scenarios()