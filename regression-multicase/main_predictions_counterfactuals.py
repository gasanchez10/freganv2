import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from numpy import corrcoef
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_data"  # Options: "synthetic_data", "student_performance", "crime"

def generate_prediction_counterfactuals(dataset_name):
    """Generate prediction counterfactuals from saved fair_model predictions"""
    
    # Load original data
    version = "1.0.0"
    data_path = f"../{dataset_name}/data/"
    
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").iloc[:, 0].values
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").iloc[:, 0].values
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").iloc[:, 0].values
    
    # Create results folder
    pred_cf_folder = f"./{dataset_name}_predictions_counterfactuals"
    os.makedirs(pred_cf_folder, exist_ok=True)
    
    print(f"=== PREDICTION COUNTERFACTUALS: {dataset_name.upper()} ===")
    
    # Process each alpha value
    alpha_range = np.arange(0, 1.1, 0.1)
    all_pred_cf = {}
    
    for a_f in alpha_range:
        # Load predictions from both baseline and CATE models in dataset folder
        dataset_pred_folder = f"./{dataset_name}_predictions"
        baseline_folder = f"{dataset_pred_folder}/Baseline_predictions_alpha_{a_f:.1f}"
        cate_folder = f"{dataset_pred_folder}/CATE_predictions_alpha_{a_f:.1f}"
        
        try:
            # Load both baseline and CATE predictions
            baseline_preds = pd.read_csv(f"{baseline_folder}/test_predictions.csv")['predictions'].values
            cate_preds = pd.read_csv(f"{cate_folder}/test_predictions.csv")['predictions'].values
            
            # Generate counterfactual predictions by using the opposite model
            pred_cf = []
            
            # Generate counterfactual predictions using CATE difference
            for i in range(len(test_t)):
                if test_t[i] == 1:  # If treated, counterfactual is control outcome
                    # Use CATE predictions with treatment effect removed
                    pred_cf.append(cate_preds[i] - (cate_preds[i] - baseline_preds[i]))
                else:  # If control, counterfactual is treatment outcome  
                    # Use CATE predictions with treatment effect added
                    pred_cf.append(cate_preds[i] + (cate_preds[i] - baseline_preds[i]))
            
            pred_cf = np.array(pred_cf)
            all_pred_cf[a_f] = pred_cf
            
            # Create potential predictions structure [Y0_pred, Y1_pred]
            potential_preds = []
            for i in range(len(test_t)):
                if test_t[i] == 0:  # Control
                    potential_preds.append([baseline_preds[i], pred_cf[i]])  # [Y0_pred, Y1_pred]
                else:  # Treatment
                    potential_preds.append([pred_cf[i], baseline_preds[i]])  # [Y0_pred, Y1_pred]
            
            potential_preds = np.array(potential_preds)
            
            # Save results for this alpha
            alpha_folder = os.path.join(pred_cf_folder, f"alpha_{a_f:.1f}")
            os.makedirs(alpha_folder, exist_ok=True)
            
            # Save prediction counterfactuals
            pd.DataFrame(pred_cf, columns=['pred_counterfactual']).to_csv(
                f"{alpha_folder}/test_pred_cf_{version}_continuous_alpha{a_f:.1f}.csv", index=False)
            
            # Save potential predictions
            pd.DataFrame(potential_preds, columns=['Y0_pred', 'Y1_pred']).to_csv(
                f"{alpha_folder}/test_potential_predictions_{version}_continuous_alpha{a_f:.1f}.csv", index=False)
            
            # Calculate correlation with actual outcomes
            correlation = corrcoef(test_y, pred_cf)[0][1] if np.std(pred_cf) > 0 else np.nan
            
            print(f"Alpha {a_f:.1f}: Correlation = {correlation:.4f}")
            
        except FileNotFoundError:
            print(f"Warning: Predictions not found for alpha {a_f:.1f}")
            continue
    
    # Generate summary correlation plot
    if all_pred_cf:
        plt.figure(figsize=(12, 8))
        
        # Plot correlations for different alphas
        alphas = list(all_pred_cf.keys())
        correlations = []
        
        for a_f in alphas:
            pred_cf = all_pred_cf[a_f]
            corr = corrcoef(test_y, pred_cf)[0][1] if np.std(pred_cf) > 0 else np.nan
            correlations.append(corr)
        
        plt.subplot(2, 2, 1)
        plt.plot(alphas, correlations, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Alpha')
        plt.ylabel('Correlation')
        plt.title('Prediction CF Correlation vs Alpha')
        plt.grid(True, alpha=0.3)
        
        # Plot example scatter for best alpha
        best_alpha = alphas[np.nanargmax(correlations)]
        best_pred_cf = all_pred_cf[best_alpha]
        
        plt.subplot(2, 2, 2)
        plt.scatter(test_y, best_pred_cf, alpha=0.6)
        plt.xlabel('Actual Outcomes')
        plt.ylabel('Predicted Counterfactuals')
        plt.title(f'Best Alpha {best_alpha:.1f} (r={correlations[np.nanargmax(correlations)]:.3f})')
        plt.grid(True, alpha=0.3)
        
        # Add line of best fit
        if not np.isnan(correlations[np.nanargmax(correlations)]):
            z = np.polyfit(test_y, best_pred_cf, 1)
            p = np.poly1d(z)
            plt.plot(test_y, p(test_y), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"{pred_cf_folder}/prediction_counterfactuals_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary results
        with open(f"{pred_cf_folder}/summary_results.txt", "w") as f:
            f.write(f"Prediction Counterfactuals Summary - {dataset_name}\\n")
            f.write(f"Best Alpha: {best_alpha:.1f}\\n")
            f.write(f"Best Correlation: {max(correlations):.4f}\\n")
            f.write(f"Treatment prevalence: {np.mean(test_t):.3f}\\n")
        
        print(f"\\nBest results: Alpha {best_alpha:.1f}, Correlation {max(correlations):.4f}")
        print(f"Results saved to: {pred_cf_folder}")
    
    return all_pred_cf

def main():
    """Generate prediction counterfactuals for specified dataset"""
    print(f"=== PREDICTION COUNTERFACTUALS ANALYSIS: {DATASET.upper()} ===")
    
    # Generate prediction counterfactuals
    pred_cf_results = generate_prediction_counterfactuals(DATASET)
    
    print(f"- Prediction counterfactuals generated successfully!")
    
    return pred_cf_results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_data", "student_performance", "crime"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python main_predictions_counterfactuals.py [dataset_name]")
        sys.exit(1)
    
    main()