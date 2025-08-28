import pandas as pd
import numpy as np
import os
from glob import glob

def evaluate_ganite_alpha_quality():
    """
    Programmatically identify the best GANITE alpha based on realistic observable criteria:
    1. MSE performance and improvement potential
    2. ATE stability across alpha values
    3. CATE reasonableness (not extreme values)
    4. Overall consistency and convergence behavior
    """
    
    # Load fair model results
    fair_results = pd.read_csv('./fair_model_average_results.csv')
    
    # Load fairness analysis results
    fairness_results = pd.read_csv('./fairness_results/complete_fairness_results.csv')
    
    # Filter GANITE results only
    ganite_columns = [col for col in fair_results.columns if 'ganite' in col.lower()]
    ganite_fairness = fairness_results[fairness_results['dataset_type'].str.contains('ganite', case=False)]
    
    # Extract GANITE alpha numbers
    ganite_alphas = []
    for col in ganite_columns:
        alpha_num = col.split('_α')[1] if '_α' in col else col.split('_alpha')[1]
        ganite_alphas.append(int(alpha_num))
    
    ganite_alphas = sorted(list(set(ganite_alphas)))
    
    results = []
    
    for alpha in ganite_alphas:
        # 1. MSE Performance Criteria
        mse_col = f'mse_ganite_α{alpha}'
        if mse_col in fair_results.columns:
            # Best MSE across all alpha values
            best_mse = fair_results[mse_col].min()
            # MSE at alpha=0.0 (pure counterfactual)
            mse_at_0 = fair_results[mse_col].iloc[0]  # alpha=0.0 is first row
            # MSE improvement ratio
            mse_improvement = (mse_at_0 - best_mse) / mse_at_0 if mse_at_0 > 0 else 0
        else:
            best_mse, mse_at_0, mse_improvement = float('inf'), float('inf'), 0
        
        # 2. ATE Stability Criteria (observable only)
        alpha_fairness = ganite_fairness[ganite_fairness['dataset_type'] == f'ganite_α{alpha}']
        if not alpha_fairness.empty:
            ate_values = alpha_fairness['ate'].values
            ate_mean = np.mean(ate_values)
            ate_std = np.std(ate_values)  # Lower std = more stable
            ate_range = np.max(ate_values) - np.min(ate_values)
            
            # Check if ATE is positive (assuming treatment should be beneficial)
            ate_positivity = 1.0 if ate_mean > 0 else 0.0
        else:
            ate_mean, ate_std, ate_range, ate_positivity = 0, float('inf'), float('inf'), 0
        
        # 3. CATE Reasonableness Criteria (observable only)
        if not alpha_fairness.empty:
            cate_values = alpha_fairness['cate_mean'].values
            cate_mean = np.mean(cate_values)
            cate_std = np.std(cate_values)  # Lower std = more stable
            cate_range = np.max(cate_values) - np.min(cate_values)
            
            # Penalize extremely high CATE values (likely noise)
            cate_reasonableness = 1.0 / (1.0 + cate_mean / 100000)  # Normalize by 100k
        else:
            cate_mean, cate_std, cate_range, cate_reasonableness = 0, float('inf'), float('inf'), 0
        
        # 4. Composite Score Calculation (observable metrics only)
        # Normalize metrics
        mse_score = 1 / (1 + best_mse)  # Higher is better, so invert
        ate_stability_score = 1 / (1 + ate_std)  # Lower std is better
        cate_stability_score = 1 / (1 + cate_std)  # Lower std is better
        improvement_score = max(0, mse_improvement)  # Higher improvement is better
        
        # Weighted composite score (realistic criteria)
        composite_score = (
            0.35 * mse_score +           # 35% weight on MSE performance
            0.25 * ate_stability_score + # 25% weight on ATE stability
            0.2 * cate_stability_score + # 20% weight on CATE stability
            0.1 * cate_reasonableness +  # 10% weight on CATE reasonableness
            0.1 * improvement_score      # 10% weight on improvement potential
        )
        
        results.append({
            'ganite_alpha': alpha,
            'best_mse': best_mse,
            'mse_at_alpha_0': mse_at_0,
            'mse_improvement_ratio': mse_improvement,
            'ate_mean': ate_mean,
            'ate_std': ate_std,
            'ate_positivity': ate_positivity,
            'cate_mean': cate_mean,
            'cate_std': cate_std,
            'cate_reasonableness': cate_reasonableness,
            'composite_score': composite_score
        })
    
    # Convert to DataFrame and sort by composite score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('composite_score', ascending=False)
    
    # Save results
    results_df.to_csv('./ganite_alpha_evaluation.csv', index=False)
    
    # Get best alpha
    best_alpha = results_df.iloc[0]['ganite_alpha']
    best_score = results_df.iloc[0]['composite_score']
    
    # Create summary report
    with open('./best_ganite_alpha_report.txt', 'w') as f:
        f.write("GANITE Alpha Selection Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"RECOMMENDED GANITE ALPHA: {int(best_alpha)}\n")
        f.write(f"Composite Score: {best_score:.4f}\n\n")
        
        f.write("RANKING OF ALL ALPHAS:\n")
        f.write("-" * 30 + "\n")
        for idx, row in results_df.iterrows():
            f.write(f"Alpha {int(row['ganite_alpha'])}: Score = {row['composite_score']:.4f}\n")
            f.write(f"  - Best MSE: {row['best_mse']:.4f}\n")
            f.write(f"  - ATE Std: {row['ate_std']:.4f}\n")
            f.write(f"  - CATE Reasonableness: {row['cate_reasonableness']:.4f}\n")
            f.write(f"  - MSE Improvement: {row['mse_improvement_ratio']:.4f}\n\n")
        
        f.write("SELECTION CRITERIA (Observable Metrics Only):\n")
        f.write("-" * 45 + "\n")
        f.write("1. MSE Performance (35%): Lower MSE across alpha values\n")
        f.write("2. ATE Stability (25%): Consistent treatment effect estimates\n")
        f.write("3. CATE Stability (20%): Stable individual treatment effects\n")
        f.write("4. CATE Reasonableness (10%): Avoids extreme CATE values\n")
        f.write("5. Improvement Potential (10%): Good performance gains possible\n\n")
        
        f.write("USAGE:\n")
        f.write("-" * 10 + "\n")
        f.write(f"Use GANITE with alpha={int(best_alpha)} for optimal fairness-performance trade-off\n")
        f.write("This alpha provides the best balance of predictive accuracy and causal inference quality.\n")
        f.write("No ground truth counterfactuals needed - based on observable metrics only.\n")
    
    print(f"✅ Best GANITE Alpha: {int(best_alpha)} (Score: {best_score:.4f})")
    print(f"📊 Full evaluation saved to: ganite_alpha_evaluation.csv")
    print(f"📋 Report saved to: best_ganite_alpha_report.txt")
    
    return int(best_alpha), best_score, results_df

if __name__ == "__main__":
    best_alpha, score, evaluation_df = evaluate_ganite_alpha_quality()
    
    # Display top 3 recommendations
    print(f"\n🏆 TOP 3 GANITE ALPHA RECOMMENDATIONS:")
    print("-" * 40)
    for i in range(min(3, len(evaluation_df))):
        row = evaluation_df.iloc[i]
        print(f"{i+1}. Alpha {int(row['ganite_alpha'])}: Score = {row['composite_score']:.4f}")
        print(f"   MSE: {row['best_mse']:.2f}, ATE Std: {row['ate_std']:.2f}")