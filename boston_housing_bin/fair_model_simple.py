import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def fair_metrics_calculator_rf(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, alpha_range=None):
    """Calculate accuracy using Random Forest with linear combination of labels"""
    if alpha_range is None:
        alpha_range = np.arange(0, 1.1, 0.1)
    
    # Calculate probability constants
    probunos = train_t.iloc[:, 0].sum() / len(train_t)
    probceros = (len(train_t) - train_t.iloc[:, 0].sum()) / len(train_t)
    
    fact_const = []
    coun_const = []
    for i in train_t.iloc[:, 0]:
        if i == 1:
            fact_const.append(probunos)
            coun_const.append(probceros)
        else:
            fact_const.append(probceros)
            coun_const.append(probunos)
    
    accuracy_results = {}
    
    for a_f in alpha_range:
        print(f"  Training Random Forest for alpha {a_f:.1f}")
        
        # Create linear combination of labels
        y_combined_labels = a_f * train_y_factual.values.ravel() + (1 - a_f) * (train_y_counterfactual.values.ravel() * np.array(coun_const) + train_y_factual.values.ravel() * np.array(fact_const))
        y_combined_labels = np.round(np.clip(y_combined_labels, 0, 1)).astype(int)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(train_x.values, y_combined_labels)
        
        # Predict
        test_pred = rf.predict(test_x.values)
        accuracy = accuracy_score(test_y.values, test_pred)
        accuracy_results[a_f] = accuracy
        
        print(f"Alpha: {a_f:.1f}, Accuracy: {accuracy:.4f}")
    
    return accuracy_results

def load_cate_counterfactuals():
    """Load CATE counterfactuals"""
    try:
        train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_binary_cate.csv")
        train_t = pd.read_csv("./data/train_t_1.0.0_binary.csv")
        
        train_cf_cate = []
        for i in range(len(train_potential)):
            if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
                train_cf_cate.append(train_potential.iloc[i, 1])
            else:  # Treatment group, counterfactual is Y0
                train_cf_cate.append(train_potential.iloc[i, 0])
        
        return pd.DataFrame(train_cf_cate, columns=['0'])
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return None

def run_simple_fair_model():
    """Run simple fair model analysis with Random Forest"""
    version = "1.0.0"
    base_path = "./data/"
    
    train_x = pd.read_csv(f"{base_path}train_x_{version}_binary.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_binary.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_binary.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_binary.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_binary.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_binary.csv")
    
    # Load CATE counterfactuals
    cate_cf = load_cate_counterfactuals()
    if cate_cf is None:
        print("Failed to load CATE counterfactuals")
        return
    
    print("CATE counterfactuals loaded successfully")
    print(f"Factual labels range: {train_y.values.min():.3f} - {train_y.values.max():.3f}")
    print(f"CATE counterfactuals range: {cate_cf.values.min():.3f} - {cate_cf.values.max():.3f}")
    
    # Run analysis
    accuracy_results = fair_metrics_calculator_rf(train_x, train_y, cate_cf, test_x, test_y, train_t, test_t)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    alphas = list(accuracy_results.keys())
    accuracies = list(accuracy_results.values())
    
    plt.plot(alphas, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Fair Model Performance - Boston Housing (CATE, Random Forest)', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("fair_model_simple_rf.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_df = pd.DataFrame({'alpha': alphas, 'accuracy': accuracies})
    results_df.to_csv("fair_model_simple_rf_results.csv", index=False)
    
    print(f"\nResults saved to fair_model_simple_rf_results.csv")
    print(f"Plot saved to fair_model_simple_rf.png")
    print(f"Best accuracy: {max(accuracies):.4f} at alpha {alphas[accuracies.index(max(accuracies))]:.1f}")

if __name__ == "__main__":
    run_simple_fair_model()