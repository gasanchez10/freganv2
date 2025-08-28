import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from numpy import corrcoef

def doubly_robust_cate(X, y, treatment_col='T'):
    """Calculate CATE using doubly robust estimation for binary classification."""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features, T)
    ps_proba = ps_model.predict_proba(X_features)
    ps = ps_proba[:, 1] if ps_proba.shape[1] > 1 else ps_proba[:, 0]
    
    # Estimate outcome models with error handling
    mu0_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==0], y[T==0])
    mu0_proba = mu0_model.predict_proba(X_features)
    mu0 = mu0_proba[:, 1] if mu0_proba.shape[1] > 1 else mu0_proba[:, 0]
    
    mu1_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_features[T==1], y[T==1])
    mu1_proba = mu1_model.predict_proba(X_features)
    mu1 = mu1_proba[:, 1] if mu1_proba.shape[1] > 1 else mu1_proba[:, 0]
    
    # Clip propensity scores to avoid division by zero
    ps = np.clip(ps, 0.01, 0.99)
    
    # Individual CATE estimates using doubly robust method
    try:
        cate_dr = ((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    except:
        # Fallback to regression-only CATE
        cate_dr = mu1 - mu0
    
    return {
        'cate_regression': cate_dr,
        'mu0': mu0,
        'mu1': mu1,
        'propensity_scores': ps
    }

def process_cate_scenarios():
    """Process CATE scenarios and generate counterfactuals"""
    
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
            
            print(f"Processing CATE for {case_dir}")
            
            # Prepare features for CATE estimation
            combined_x = pd.concat([train_x, test_x], ignore_index=True)
            combined_t = pd.concat([train_t, test_t], ignore_index=True)
            combined_y = pd.concat([train_y, test_y], ignore_index=True)
            
            # Check if predictions have variation
            unique_train_y = np.unique(train_y.iloc[:, 0].values)
            unique_test_y = np.unique(test_y.iloc[:, 0].values)
            
            if len(unique_train_y) == 1 and len(unique_test_y) == 1:
                print(f"Warning: All predictions are the same value ({unique_train_y[0]}). Using original data instead.")
                # Load original data
                orig_train_y = pd.read_csv("../data/train_y_1.0.0_binary.csv")
                orig_test_y = pd.read_csv("../data/test_y_1.0.0_binary.csv")
                combined_y = pd.concat([orig_train_y, orig_test_y], ignore_index=True)
                # Also update train_y and test_y for consistency and reset indices
                train_y = orig_train_y.reset_index(drop=True)
                test_y = orig_test_y.reset_index(drop=True)
            
            # Add treatment column to features
            combined_x['T'] = combined_t.iloc[:, 0].values
            combined_y_values = combined_y.iloc[:, 0].values
            
            # Calculate CATE
            cate_results = doubly_robust_cate(combined_x, combined_y_values)
            cate_estimates = cate_results['cate_regression']
            
            # Use regression-based CATE if doubly robust has NaN values
            if np.isnan(cate_estimates).any():
                cate_estimates = cate_results['mu1'] - cate_results['mu0']
            
            # Split CATE back to train/test and convert to numpy arrays
            train_size = len(train_t)
            train_cate = np.array(cate_estimates[:train_size])
            test_cate = np.array(cate_estimates[train_size:])
            
            # Generate counterfactuals using CATE
            train_counterfactuals = []
            test_counterfactuals = []
            
            # Convert to arrays for easier indexing
            train_t_arr = train_t.iloc[:, 0].values
            train_y_arr = train_y.iloc[:, 0].values
            test_t_arr = test_t.iloc[:, 0].values
            test_y_arr = test_y.iloc[:, 0].values
            
            # For training data
            for i in range(len(train_t_arr)):
                if train_t_arr[i] == 1:  # If treated (male), subtract CATE
                    train_counterfactuals.append(max(0, min(1, train_y_arr[i] - train_cate[i])))
                else:  # If control (female), add CATE
                    train_counterfactuals.append(max(0, min(1, train_y_arr[i] + train_cate[i])))
            
            # For test data
            for i in range(len(test_t_arr)):
                if test_t_arr[i] == 1:  # If treated (male), subtract CATE
                    test_counterfactuals.append(max(0, min(1, test_y_arr[i] - test_cate[i])))
                else:  # If control (female), add CATE
                    test_counterfactuals.append(max(0, min(1, test_y_arr[i] + test_cate[i])))
            
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
            
            for i in range(len(train_t_arr)):
                if train_t_arr[i] == 0:  # Control (female)
                    train_potential_y.append([train_y_arr[i], train_counterfactuals[i]])  # [Y0, Y1]
                else:  # Treatment (male)
                    train_potential_y.append([train_counterfactuals[i], train_y_arr[i]])  # [Y0, Y1]
            
            for i in range(len(test_t_arr)):
                if test_t_arr[i] == 0:  # Control (female)
                    test_potential_y.append([test_y_arr[i], test_counterfactuals[i]])  # [Y0, Y1]
                else:  # Treatment (male)
                    test_potential_y.append([test_counterfactuals[i], test_y_arr[i]])  # [Y0, Y1]
            
            # Save potential outcomes
            train_potential_df = pd.DataFrame(train_potential_y, columns=['0', '1'])
            test_potential_df = pd.DataFrame(test_potential_y, columns=['0', '1'])
            
            train_potential_df.to_csv(os.path.join(results_folder, f"train_potential_y_{version}_binary_cate.csv"), index=False)
            test_potential_df.to_csv(os.path.join(results_folder, f"test_potential_y_{version}_binary_cate.csv"), index=False)
            
            # Create test_y_hat (combined train and test potential outcomes)
            combined_potential_y = pd.concat([train_potential_df, test_potential_df], ignore_index=True)
            combined_potential_y.to_csv(os.path.join(results_folder, f"test_y_hat_{version}_binary_cate.csv"), index=False)
            
            # Generate correlation plot
            correlation = corrcoef(train_y_arr, train_counterfactuals)[0][1]
            
            plt.figure(figsize=(8, 6))
            plt.xlabel('Train Predictions')
            plt.ylabel('CATE Counterfactuals')
            plt.title(f'CATE - Correlation: {correlation:.3f}')
            plt.scatter(train_y_arr, train_counterfactuals, alpha=0.6)
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
            import traceback
            print(f"Error processing {case_dir}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue

if __name__ == '__main__':
    process_cate_scenarios()