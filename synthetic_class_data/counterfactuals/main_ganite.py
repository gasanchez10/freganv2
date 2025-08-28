import argparse
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from numpy import corrcoef

from ganite import ganite
from data_loading import data_loading_synthetic

def main(args):
    """Train GANITE across multiple alpha values and evaluate performance."""
    version = "1.0.0"
    base_path = "./"
    # Test different alpha values for loss balancing
    alphas = [5, 6, 7, 8]
    optimal_alphas = []

    for i in alphas:
        # Load synthetic causal data
        train_x, train_t, train_y, train_potential_y, test_x, test_potential_y, test_t, treatment, test_y, train_cf_manual = data_loading_synthetic()
        
        # Configure GANITE parameters
        parameters = {
            'h_dim': args.h_dim,
            'iteration': args.iteration,
            'batch_size': args.batch_size,
            'alpha': i  # Loss balance parameter
        }
        
        # Train GANITE model
        test_y_hat = ganite(train_x, train_t, train_y, test_x, test_y, test_t, parameters)

        # Save all data and results to CSV files in alpha-specific folders
        alpha_folder = os.path.join(base_path, f"alpha_{i}")
        os.makedirs(alpha_folder, exist_ok=True)
        
        data_arrays = [train_x, train_t, train_y, train_potential_y, test_y, test_t, test_x, test_t, test_y, test_potential_y, test_y_hat]
        data_names = ["train_x", "train_t", "train_y", "train_potential_y", "test_y", "test_t", "test_x", "test_t", "test_y", "test_potential_y", "test_y_hat"]
        for name, data in zip(data_names, data_arrays):
            df = pd.DataFrame(data)
            save_path = os.path.join(alpha_folder, f"{name}_{version}_binary_ganite_alpha{i}.csv")
            df.to_csv(save_path, index=False)

        # Extract counterfactual outcomes for training data
        train_y_hat_ganite = test_y_hat[0:len(train_t)]
        test_y_hat_ganite=test_y_hat[len(train_t):len(train_t)+len(test_t)]

        train_cf_ganite = []
        for k in range(min(len(train_t), len(train_y_hat_ganite))):
            # Get counterfactual outcome (opposite treatment)
            train_cf_ganite.append(train_y_hat_ganite[k][1-int(train_t.iloc[k, 0])])
        train_cf_ganite = np.array(train_cf_ganite)

        # Generate correlation plot for all alpha values
        train_cf_manual_values = train_cf_manual.iloc[:len(train_cf_ganite), 0].values
        correlation = corrcoef(train_cf_manual_values, train_cf_ganite)[0][1]
        
        plt.figure(figsize=(8, 6))
        plt.xlabel('CF Manual')
        plt.ylabel('CF GANITE')
        plt.title(f'Alpha {i} - Correlation: {correlation:.3f}')
        plt.scatter(train_cf_manual_values, train_cf_ganite, alpha=0.6)
        plt.grid(True, alpha=0.3)
        save_path_graph = os.path.join(alpha_folder, f"correlation_alpha_{i}.png")
        plt.savefig(save_path_graph, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Track optimal alpha values (correlation > 0.7)
        if correlation > 0.7:
            optimal_alphas.append((i, correlation))
    
    # Save optimal alpha values to file
    with open(os.path.join(base_path, "optimal_alpha_ganite.txt"), "w") as f:
        f.write("Optimal Alpha Values (Correlation > 0.7):\n")
        f.write("Alpha\tCorrelation\n")
        for alpha, corr in optimal_alphas:
            f.write(f"{alpha}\t{corr:.4f}\n")
        if not optimal_alphas:
            f.write("No alpha values achieved correlation > 0.7\n")
    
    return test_y_hat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_dim', default=30, type=int, help='Hidden layer dimensions')
    parser.add_argument('--iteration', default=10000, type=int, help='Training iterations')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    
    args = parser.parse_args()
    test_y_hat = main(args)