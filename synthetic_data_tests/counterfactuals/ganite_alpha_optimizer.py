import argparse
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from numpy import corrcoef

from ganite import ganite
from data_loading import data_loading_synthetic

def evaluate_ganite_alpha_during_training(alpha_range=[0, 1, 2, 3, 4, 5]):
    """
    Evaluate GANITE alpha values during training using only observable metrics:
    1. Training loss convergence
    2. Counterfactual consistency (self-consistency check)
    3. Treatment effect magnitude reasonableness
    4. Correlation with manual counterfactuals (if available)
    """
    
    version = "1.0.0"
    base_path = "./"
    
    # Load data once
    train_x, train_t, train_y, train_potential_y, test_x, test_potential_y, test_t, treatment, test_y, train_cf_manual = data_loading_synthetic()
    
    evaluation_results = []
    
    for alpha in alpha_range:
        print(f"\n{'='*50}")
        print(f"EVALUATING GANITE ALPHA: {alpha}")
        print(f"{'='*50}")
        
        # Configure GANITE parameters
        parameters = {
            'h_dim': 30,
            'iteration': 10000,
            'batch_size': 256,
            'alpha': alpha
        }
        
        # Train GANITE model
        test_y_hat = ganite(train_x, train_t, train_y, test_x, test_y, test_t, parameters)
        
        # Extract counterfactuals for training data
        train_y_hat_ganite = test_y_hat[0:len(train_t)]
        
        train_cf_ganite = []
        for k in range(min(len(train_t), len(train_y_hat_ganite))):
            cf_treatment = 1 - int(train_t.iloc[k, 0])
            train_cf_ganite.append(train_y_hat_ganite[k][cf_treatment])
        train_cf_ganite = np.array(train_cf_ganite)
        
        # 1. Self-consistency check: factual predictions should match actual outcomes
        factual_predictions = []
        for k in range(len(train_t)):
            actual_treatment = int(train_t.iloc[k, 0])
            factual_predictions.append(train_y_hat_ganite[k][actual_treatment])
        factual_predictions = np.array(factual_predictions)
        
        factual_mse = np.mean((factual_predictions - train_y.values.ravel())**2)
        factual_consistency = 1.0 / (1.0 + factual_mse)  # Higher is better
        
        # 2. Treatment effect magnitude check
        treatment_effects = []
        for k in range(len(train_t)):
            te = train_y_hat_ganite[k][1] - train_y_hat_ganite[k][0]  # T=1 - T=0
            treatment_effects.append(te)
        treatment_effects = np.array(treatment_effects)
        
        ate_estimate = np.mean(treatment_effects)
        ate_std = np.std(treatment_effects)
        
        # Reasonableness: ATE should be positive and not extreme
        ate_reasonableness = 1.0 if ate_estimate > 0 else 0.5
        if abs(ate_estimate) > 100:  # Penalize extreme values
            ate_reasonableness *= 0.5
        
        # 3. Counterfactual stability (variance across individuals)
        cf_stability = 1.0 / (1.0 + np.std(train_cf_ganite))
        
        # 4. Correlation with manual counterfactuals (if available)
        if len(train_cf_manual) > 0:
            train_cf_manual_values = train_cf_manual.iloc[:len(train_cf_ganite), 0].values
            correlation = corrcoef(train_cf_manual_values, train_cf_ganite)[0][1]
            correlation = max(0, correlation)  # Only positive correlations count
        else:
            correlation = 0.5  # Neutral if no manual counterfactuals
        
        # 5. Composite quality score (observable metrics only)
        quality_score = (
            0.3 * factual_consistency +    # 30% - Model should predict factuals well
            0.25 * ate_reasonableness +    # 25% - Treatment effect should be reasonable
            0.2 * cf_stability +           # 20% - Counterfactuals should be stable
            0.15 * correlation +           # 15% - Correlation with manual (if available)
            0.1 * (1.0 / (1.0 + ate_std))  # 10% - Treatment effects should be consistent
        )
        
        evaluation_results.append({
            'alpha': alpha,
            'factual_mse': factual_mse,
            'factual_consistency': factual_consistency,
            'ate_estimate': ate_estimate,
            'ate_std': ate_std,
            'ate_reasonableness': ate_reasonableness,
            'cf_stability': cf_stability,
            'correlation_with_manual': correlation,
            'quality_score': quality_score
        })
        
        print(f"Factual MSE: {factual_mse:.4f}")
        print(f"ATE Estimate: {ate_estimate:.4f}")
        print(f"Correlation with Manual: {correlation:.4f}")
        print(f"Quality Score: {quality_score:.4f}")
        
        # Save this alpha's results
        alpha_folder = os.path.join(base_path, f"alpha_{alpha}")
        os.makedirs(alpha_folder, exist_ok=True)
        
        # Save all required data
        data_arrays = [train_x, train_t, train_y, train_potential_y, test_y, test_t, test_x, test_t, test_y, test_potential_y, test_y_hat]
        data_names = ["train_x", "train_t", "train_y", "train_potential_y", "test_y", "test_t", "test_x", "test_t", "test_y", "test_potential_y", "test_y_hat"]
        for name, data in zip(data_names, data_arrays):
            df = pd.DataFrame(data)
            save_path = os.path.join(alpha_folder, f"{name}_{version}_continous_ganite_alpha{alpha}.csv")
            df.to_csv(save_path, index=False)
    
    # Convert results to DataFrame and find best alpha
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values('quality_score', ascending=False)
    
    best_alpha = results_df.iloc[0]['alpha']
    best_score = results_df.iloc[0]['quality_score']
    
    # Save evaluation results
    results_df.to_csv(os.path.join(base_path, "ganite_alpha_training_evaluation.csv"), index=False)
    
    # Create optimal alpha summary
    with open(os.path.join(base_path, "optimal_alpha_ganite.txt"), "w") as f:
        f.write("GANITE Alpha Training-Time Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(f"OPTIMAL ALPHA: {int(best_alpha)}\n")
        f.write(f"Quality Score: {best_score:.4f}\n\n")
        
        f.write("EVALUATION CRITERIA (Training-Time Observable):\n")
        f.write("-" * 45 + "\n")
        f.write("1. Factual Consistency (30%): Model predicts known outcomes well\n")
        f.write("2. ATE Reasonableness (25%): Treatment effect is positive and reasonable\n")
        f.write("3. Counterfactual Stability (20%): Stable counterfactual predictions\n")
        f.write("4. Manual Correlation (15%): Correlation with manual counterfactuals\n")
        f.write("5. Treatment Effect Consistency (10%): Consistent individual effects\n\n")
        
        f.write("RANKING:\n")
        f.write("-" * 10 + "\n")
        for idx, row in results_df.iterrows():
            f.write(f"Alpha {int(row['alpha'])}: Score = {row['quality_score']:.4f}\n")
    
    print(f"\n🏆 OPTIMAL GANITE ALPHA: {int(best_alpha)} (Score: {best_score:.4f})")
    print(f"📊 Evaluation saved to: ganite_alpha_training_evaluation.csv")
    
    return int(best_alpha), results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_range', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], 
                       help='Range of alpha values to evaluate')
    
    args = parser.parse_args()
    optimal_alpha, evaluation_df = evaluate_ganite_alpha_during_training(args.alpha_range)