import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from numpy import corrcoef
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

def generate_predictions_counterfactuals_cate(dataset_name):
    """Generate prediction counterfactuals using CATE approach for classification datasets"""
    
    print(f"=== PREDICTION COUNTERFACTUALS ANALYSIS: {dataset_name.upper()} ===")
    
    # Define paths
    predictions_dir = f"./{dataset_name}_predictions"
    cate_results_dir = f"./{dataset_name}_cate_results"
    output_dir = f"./{dataset_name}_predictions_counterfactuals"
    os.makedirs(output_dir, exist_ok=True)
    
    alpha_range = np.arange(0, 1.1, 0.1)
    
    # Check if required directories exist
    if not os.path.exists(predictions_dir):
        print(f"Error: Predictions directory {predictions_dir} not found!")
        print("Please run fair_model.py first to generate predictions.")
        return {}
    
    if not os.path.exists(cate_results_dir):
        print(f"Error: CATE results directory {cate_results_dir} not found!")
        print("Please run main_cate_counterfactuals.py first to generate CATE counterfactuals.")
        return {}
    
    # Load CATE counterfactuals
    try:
        if dataset_name == "synthetic_class_data":
            file_suffix = "binary_cate"
        else:  # german_credit
            file_suffix = "continuous_cate"
        
        train_potential = pd.read_csv(f"{cate_results_dir}/train_potential_y_1.0.0_{file_suffix}.csv")
        train_t = pd.read_csv(f"{cate_results_dir}/train_t_1.0.0_{file_suffix}.csv")
        test_potential = pd.read_csv(f"{cate_results_dir}/test_potential_y_1.0.0_{file_suffix}.csv")
        test_t = pd.read_csv(f"{cate_results_dir}/test_t_1.0.0_{file_suffix}.csv")
        
        print(f"Loaded CATE counterfactuals: {len(train_potential)} train, {len(test_potential)} test")
        
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return {}
    
    # Store correlation results for each alpha
    correlation_results = {}
    
    print(f"=== PREDICTION COUNTERFACTUALS: {dataset_name.upper()} ===")
    
    for alpha_val in alpha_range:
        alpha_str = f"alpha_{alpha_val:.1f}"
        case_name = f"CATE_predictions_{alpha_str}"
        
        case_path = os.path.join(predictions_dir, case_name)
        
        if not os.path.exists(case_path):
            print(f"Case {case_name} not found, skipping...")
            continue
        
        # Check if predictions file exists
        if not os.path.exists(os.path.join(case_path, "test_predictions.csv")):
            print(f"Missing test_predictions.csv in {case_name}, skipping...")
            continue
        
        try:
            # Load predictions from fair_model
            test_predictions = pd.read_csv(os.path.join(case_path, "test_predictions.csv"))
            
            # Generate prediction counterfactuals using CATE approach
            train_pred_counterfactuals = []
            test_pred_counterfactuals = []
            
            # For training data
            for i in range(len(train_potential)):
                if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
                    train_pred_counterfactuals.append(train_potential.iloc[i, 1])
                else:  # Treatment group, counterfactual is Y0
                    train_pred_counterfactuals.append(train_potential.iloc[i, 0])
            
            # For test data
            for i in range(len(test_potential)):
                if test_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
                    test_pred_counterfactuals.append(test_potential.iloc[i, 1])
                else:  # Treatment group, counterfactual is Y0
                    test_pred_counterfactuals.append(test_potential.iloc[i, 0])
            
            train_pred_counterfactuals = np.array(train_pred_counterfactuals)
            test_pred_counterfactuals = np.array(test_pred_counterfactuals)
            
            # Calculate correlation with test predictions
            test_pred_values = test_predictions.iloc[:, 0].values
            
            if len(np.unique(test_pred_values)) > 1 and len(np.unique(test_pred_counterfactuals)) > 1:
                correlation = corrcoef(test_pred_values, test_pred_counterfactuals)[0][1]
            else:
                correlation = 0.0
            
            correlation_results[alpha_val] = correlation
            
            # Save prediction counterfactuals for this alpha
            output_alpha_dir = os.path.join(output_dir, alpha_str)
            os.makedirs(output_alpha_dir, exist_ok=True)
            
            version = "1.0.0"
            file_suffix_save = "binary2025" if dataset_name == "synthetic_class_data" else "continuous"
            
            # Save train prediction counterfactuals
            train_pred_cf_df = pd.DataFrame({
                'pred_counterfactual': train_pred_counterfactuals
            })
            train_pred_cf_df.to_csv(
                os.path.join(output_alpha_dir, f"train_pred_cf_{version}_{file_suffix_save}_alpha{alpha_val:.1f}.csv"), 
                index=False
            )
            
            # Save test prediction counterfactuals
            test_pred_cf_df = pd.DataFrame({
                'pred_counterfactual': test_pred_counterfactuals
            })
            test_pred_cf_df.to_csv(
                os.path.join(output_alpha_dir, f"test_pred_cf_{version}_{file_suffix_save}_alpha{alpha_val:.1f}.csv"), 
                index=False
            )
            
            print(f"Alpha {alpha_val:.1f}: Correlation = {correlation:.4f}")
        
        except Exception as e:
            print(f"Error processing alpha {alpha_val:.1f}: {e}")
            continue
    
    # Generate summary plot and results
    if correlation_results:
        plt.figure(figsize=(10, 6))
        alphas = list(correlation_results.keys())
        correlations = list(correlation_results.values())
        
        plt.plot(alphas, correlations, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Alpha (Linear Combination Factor)')
        plt.ylabel('Correlation with Predictions')
        plt.title(f'{dataset_name.replace("_", " ").title()} - Prediction Counterfactuals Correlation')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add correlation values as text
        for alpha, corr in zip(alphas, correlations):
            plt.text(alpha, corr + 0.02, f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_predictions_counterfactuals_correlation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation results
        results_df = pd.DataFrame({
            'alpha': alphas,
            'correlation': correlations
        })
        results_df.to_csv(os.path.join(output_dir, f'{dataset_name}_predictions_counterfactuals_results.csv'), index=False)
        
        # Find best correlation
        best_alpha = max(correlation_results, key=correlation_results.get)
        best_correlation = correlation_results[best_alpha]
        
        print(f"\\nBest results: Alpha {best_alpha:.1f}, Correlation {best_correlation:.4f}")
        print(f"Results saved to: {output_dir}")
        print(f"- Prediction counterfactuals generated successfully!")
        
        return correlation_results
    else:
        print("No correlation results generated!")
        return {}

def main():
    """Generate prediction counterfactuals using CATE for classification datasets"""
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        return
    
    # Generate prediction counterfactuals
    results = generate_predictions_counterfactuals_cate(DATASET)
    
    return results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python main_predictions_counterfactuals.py [dataset_name]")
        sys.exit(1)
    
    main()