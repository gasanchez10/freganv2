import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import corrcoef

def process_cate_scenarios():
    """Process CATE scenarios and generate counterfactuals using simple approach"""
    
    predictions_dir = "../predictions"
    output_dir = "./"
    
    # Only process folders that start with 'cate'
    case_dirs = [d for d in os.listdir(predictions_dir) if d.startswith('cate') and os.path.isdir(os.path.join(predictions_dir, d))]
    
    for case_dir in case_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {case_dir}")
        print(f"{'='*60}")
        
        case_path = os.path.join(predictions_dir, case_dir)
        output_case_path = os.path.join(output_dir, case_dir)
        os.makedirs(output_case_path, exist_ok=True)
        
        # Check if required files exist
        required_files = [
            "train_x_1.0.0_binary.csv",
            "train_t_1.0.0_binary.csv", 
            "train_y_1.0.0_binary.csv",
            "test_x_1.0.0_binary.csv",
            "test_t_1.0.0_binary.csv",
            "test_y_1.0.0_binary.csv"
        ]
        
        if not all(os.path.exists(os.path.join(case_path, f)) for f in required_files):
            print(f"Missing required files in {case_dir}, skipping...")
            continue
        
        try:
            # Load data for this case
            train_x = pd.read_csv(os.path.join(case_path, "train_x_1.0.0_binary.csv"))
            train_t = pd.read_csv(os.path.join(case_path, "train_t_1.0.0_binary.csv"))
            train_y = pd.read_csv(os.path.join(case_path, "train_y_1.0.0_binary.csv"))
            test_x = pd.read_csv(os.path.join(case_path, "test_x_1.0.0_binary.csv"))
            test_t = pd.read_csv(os.path.join(case_path, "test_t_1.0.0_binary.csv"))
            test_y = pd.read_csv(os.path.join(case_path, "test_y_1.0.0_binary.csv"))
            
            print(f"Processing simple CATE for {case_dir}")
            
            # Check if predictions have variation
            unique_train_y = np.unique(train_y.iloc[:, 0].values)
            unique_test_y = np.unique(test_y.iloc[:, 0].values)
            
            print(f"Train predictions unique values: {unique_train_y}")
            print(f"Test predictions unique values: {unique_test_y}")
            
            # If no variation in predictions, use original data for CATE estimation
            if len(unique_train_y) == 1:
                print(f"No variation in predictions. Using original data for CATE estimation.")
                # Load original data for CATE calculation
                orig_train_y = pd.read_csv("../data/train_y_1.0.0_binary.csv")
                orig_test_y = pd.read_csv("../data/test_y_1.0.0_binary.csv")
                
                # Simple CATE estimation: difference in means by treatment group
                train_treated = train_t.iloc[:, 0] == 1
                train_control = train_t.iloc[:, 0] == 0
                
                if np.sum(train_treated) > 0 and np.sum(train_control) > 0:
                    cate_estimate = (np.mean(orig_train_y.iloc[train_treated, 0]) - 
                                   np.mean(orig_train_y.iloc[train_control, 0]))
                else:
                    cate_estimate = 0.0
                
                print(f"Simple CATE estimate: {cate_estimate:.4f}")
                
                # Generate counterfactuals using simple CATE
                train_counterfactuals = []
                test_counterfactuals = []
                
                # For training data
                for i in range(len(train_t)):
                    if train_t.iloc[i, 0] == 1:  # If treated, subtract CATE
                        cf_value = max(0, min(1, train_y.iloc[i, 0] - cate_estimate))
                    else:  # If control, add CATE
                        cf_value = max(0, min(1, train_y.iloc[i, 0] + cate_estimate))
                    train_counterfactuals.append(cf_value)
                
                # For test data
                for i in range(len(test_t)):
                    if test_t.iloc[i, 0] == 1:  # If treated, subtract CATE
                        cf_value = max(0, min(1, test_y.iloc[i, 0] - cate_estimate))
                    else:  # If control, add CATE
                        cf_value = max(0, min(1, test_y.iloc[i, 0] + cate_estimate))
                    test_counterfactuals.append(cf_value)
            
            else:
                # If there is variation, use the predictions as they are
                print(f"Using predictions with variation for counterfactuals.")
                # Simple approach: flip the predictions
                train_counterfactuals = 1 - train_y.iloc[:, 0].values
                test_counterfactuals = 1 - test_y.iloc[:, 0].values
            
            train_counterfactuals = np.array(train_counterfactuals)
            test_counterfactuals = np.array(test_counterfactuals)
            
            # Create results folder
            results_folder = os.path.join(output_case_path, "cate_results")
            os.makedirs(results_folder, exist_ok=True)
            
            # Save all data with CATE suffix
            version = "1.0.0"
            
            # Save training data
            train_x.to_csv(os.path.join(results_folder, f"train_x_{version}_binary_cate.csv"), index=False)
            train_t.to_csv(os.path.join(results_folder, f"train_t_{version}_binary_cate.csv"), index=False)
            train_y.to_csv(os.path.join(results_folder, f"train_y_{version}_binary_cate.csv"), index=False)
            
            # Save test data
            test_x.to_csv(os.path.join(results_folder, f"test_x_{version}_binary_cate.csv"), index=False)
            test_t.to_csv(os.path.join(results_folder, f"test_t_{version}_binary_cate.csv"), index=False)
            test_y.to_csv(os.path.join(results_folder, f"test_y_{version}_binary_cate.csv"), index=False)
            
            # Create potential outcomes (factual and counterfactual combined)
            train_potential_y = []
            test_potential_y = []
            
            for i in range(len(train_t)):
                if train_t.iloc[i, 0] == 0:  # Control
                    train_potential_y.append([train_y.iloc[i, 0], train_counterfactuals[i]])  # [Y0, Y1]
                else:  # Treatment
                    train_potential_y.append([train_counterfactuals[i], train_y.iloc[i, 0]])  # [Y0, Y1]
            
            for i in range(len(test_t)):
                if test_t.iloc[i, 0] == 0:  # Control
                    test_potential_y.append([test_y.iloc[i, 0], test_counterfactuals[i]])  # [Y0, Y1]
                else:  # Treatment
                    test_potential_y.append([test_counterfactuals[i], test_y.iloc[i, 0]])  # [Y0, Y1]
            
            # Save potential outcomes
            train_potential_df = pd.DataFrame(train_potential_y, columns=['0', '1'])
            test_potential_df = pd.DataFrame(test_potential_y, columns=['0', '1'])
            
            train_potential_df.to_csv(os.path.join(results_folder, f"train_potential_y_{version}_binary_cate.csv"), index=False)
            test_potential_df.to_csv(os.path.join(results_folder, f"test_potential_y_{version}_binary_cate.csv"), index=False)
            
            # Create test_y_hat (combined train and test potential outcomes)
            combined_potential_y = pd.concat([train_potential_df, test_potential_df], ignore_index=True)
            combined_potential_y.to_csv(os.path.join(results_folder, f"test_y_hat_{version}_binary_cate.csv"), index=False)
            
            # Generate correlation plot
            train_pred = train_y.iloc[:, 0].values
            if len(np.unique(train_pred)) > 1 and len(np.unique(train_counterfactuals)) > 1:
                correlation = corrcoef(train_pred, train_counterfactuals)[0][1]
            else:
                correlation = 0.0
            
            plt.figure(figsize=(8, 6))
            plt.xlabel('Train Predictions')
            plt.ylabel('CATE Counterfactuals')
            plt.title(f'CATE - Correlation: {correlation:.3f}')
            plt.scatter(train_pred, train_counterfactuals, alpha=0.6)
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(results_folder, "correlation_cate.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save correlation results
            with open(os.path.join(results_folder, "cate_correlation_results.txt"), "w") as f:
                f.write(f"CATE Correlation Results for {case_dir}\n")
                f.write(f"Correlation: {correlation:.4f}\n")
            
            print(f"Completed CATE for {case_dir} (correlation: {correlation:.3f})")
            print(f"Successfully processed {case_dir}")
                
        except Exception as e:
            print(f"Error processing {case_dir}: {e}")
            continue

if __name__ == '__main__':
    process_cate_scenarios()