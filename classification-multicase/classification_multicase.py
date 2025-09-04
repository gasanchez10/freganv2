import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

def load_dataset(dataset_name):
    """Load dataset based on configuration"""
    if dataset_name == "synthetic_class_data":
        return load_synthetic_class_data()
    elif dataset_name == "german_credit":
        return load_german_credit()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_synthetic_class_data():
    """Load synthetic classification data from saved files"""
    version = "1.0.0"
    data_path = "../synthetic_class_data/data/"
    
    # Load train and test data (using binary2025 naming convention)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_binary2025.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_binary2025.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_binary2025.csv").values.ravel()
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_binary2025.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_binary2025.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_binary2025.csv").values.ravel()
    
    # Add treatment column to features
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "Synthetic Classification", "Binary Treatment"

def load_german_credit():
    """Load German credit data from saved files (confounders-only)"""
    version = "1.0.0"
    data_path = "../german_credit/data/"
    
    # Load train and test data (confounders-only)
    train_x = pd.read_csv(f"{data_path}train_x_{version}_continuous.csv")
    train_t = pd.read_csv(f"{data_path}train_t_{version}_continuous.csv").values.ravel()
    train_y = pd.read_csv(f"{data_path}train_y_{version}_continuous.csv").values.ravel()
    
    test_x = pd.read_csv(f"{data_path}test_x_{version}_continuous.csv")
    test_t = pd.read_csv(f"{data_path}test_t_{version}_continuous.csv").values.ravel()
    test_y = pd.read_csv(f"{data_path}test_y_{version}_continuous.csv").values.ravel()
    
    # Add treatment column to features (already positive ATE with confounders-only)
    train_x['T'] = train_t
    test_x['T'] = test_t
    
    return train_x, train_y, train_t, test_x, test_y, test_t, "German Credit (Confounders)", "Gender (Male)"

def doubly_robust_ate_classification(X, y, treatment_col='T'):
    """Calculate ATE using doubly robust estimation for classification"""
    T = X[treatment_col].astype(int)
    X_features = X.drop(columns=[treatment_col])
    
    # Estimate propensity scores with clipping
    ps_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ps_model.fit(X_features, T)
    
    if len(ps_model.classes_) == 1:
        # Only one treatment class, use constant propensity
        ps = np.full(len(X_features), 0.5)
    else:
        ps = ps_model.predict_proba(X_features)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
    
    # Estimate outcome models (probabilities for classification)
    if len(X_features[T==0]) > 0 and len(X_features[T==1]) > 0:
        mu0_model = RandomForestClassifier(n_estimators=100, random_state=42)
        mu1_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        mu0_model.fit(X_features[T==0], y[T==0])
        mu1_model.fit(X_features[T==1], y[T==1])
        
        # Get probabilities for positive class
        if len(mu0_model.classes_) == 1:
            mu0 = np.full(len(X_features), mu0_model.classes_[0])
        else:
            mu0 = mu0_model.predict_proba(X_features)[:, 1]
            
        if len(mu1_model.classes_) == 1:
            mu1 = np.full(len(X_features), mu1_model.classes_[0])
        else:
            mu1 = mu1_model.predict_proba(X_features)[:, 1]
    else:
        # Fallback if one treatment group is empty
        mu0 = np.full(len(X_features), np.mean(y[T==0]) if len(y[T==0]) > 0 else 0.5)
        mu1 = np.full(len(X_features), np.mean(y[T==1]) if len(y[T==1]) > 0 else 0.5)
    
    # Doubly robust ATE (for classification - difference in probabilities)
    ate_dr = np.mean((T*(y - mu1)/ps + mu1) - ((1-T)*(y - mu0)/(1-ps) + mu0))
    ate_reg = np.mean(mu1 - mu0)
    
    return {
        'ate_doubly_robust': ate_dr,
        'ate_regression_only': ate_reg,
        'treated_fraction': np.mean(T)
    }

def run_analysis():
    """Run unified classification analysis pipeline"""
    print(f"=== UNIFIED CLASSIFICATION ANALYSIS: {DATASET.upper()} ===")
    
    # Load dataset (train and test data)
    X_train, y_train, t_train, X_test, y_test, t_test, dataset_name, treatment_name = load_dataset(DATASET)
    
    print(f"Dataset: {dataset_name}")
    print(f"Treatment: {treatment_name}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_test.shape[1]-1}")  # -1 for treatment column
    print(f"Treatment prevalence: {np.mean(t_test):.3f}")
    print(f"Positive class prevalence: {np.mean(y_test):.3f}")
    
    # Train model on training features (excluding treatment)
    X_train_features = X_train.drop('T', axis=1)
    X_test_features = X_test.drop('T', axis=1)
    
    # Use regularization parameters from synthetic_class_data fair_model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train_features, y_train)
    
    # Predictions on test set
    y_pred = rf.predict(X_test_features)
    y_pred_proba = rf.predict_proba(X_test_features)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5  # If only one class present
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Calculate ATE on full dataset
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.concatenate([y_train, y_test])
    ate_results = doubly_robust_ate_classification(X_combined, y_combined)
    
    print(f"\nCausal Analysis:")
    print(f"ATE (Doubly Robust): {ate_results['ate_doubly_robust']:.4f}")
    print(f"ATE (Regression): {ate_results['ate_regression_only']:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Confusion Matrix
    from sklearn.metrics import confusion_matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Plot 2: ROC Curve
    plt.subplot(2, 3, 2)
    from sklearn.metrics import roc_curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Only one class\nin test set', ha='center', va='center')
        plt.title('ROC Curve (N/A)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    importance = pd.Series(rf.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
    importance.head(10).plot(kind='barh')
    plt.xlabel('Importance')
    plt.title('Top 10 Features')
    
    # Plot 4: Outcome by Treatment
    plt.subplot(2, 3, 4)
    y_control = y_test[t_test == 0]
    y_treated = y_test[t_test == 1]
    
    control_pos_rate = np.mean(y_control) if len(y_control) > 0 else 0
    treated_pos_rate = np.mean(y_treated) if len(y_treated) > 0 else 0
    
    plt.bar(['Control', 'Treated'], [control_pos_rate, treated_pos_rate], alpha=0.7)
    plt.ylabel('Positive Class Rate')
    plt.title(f'Outcome by Treatment\nATE: {ate_results["ate_doubly_robust"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Prediction Probabilities
    plt.subplot(2, 3, 5)
    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Negative Class', bins=20, density=True)
    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Positive Class', bins=20, density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: ATE Summary
    plt.subplot(2, 3, 6)
    ate_methods = ['Doubly Robust', 'Regression']
    ate_values = [ate_results['ate_doubly_robust'], ate_results['ate_regression_only']]
    plt.bar(ate_methods, ate_values, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title('ATE Comparison')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(ate_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'Unified Classification Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DATASET}_unified_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'dataset': dataset_name,
        'treatment': treatment_name,
        'samples': len(X_test),
        'features': X_test.shape[1]-1,
        'treatment_prevalence': np.mean(t_test),
        'positive_class_prevalence': np.mean(y_test),
        'accuracy': accuracy,
        'auc': auc,
        'ate_doubly_robust': ate_results['ate_doubly_robust'],
        'ate_regression': ate_results['ate_regression_only']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{DATASET}_unified_classification_results.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"- {DATASET}_unified_classification_analysis.png")
    print(f"- {DATASET}_unified_classification_results.csv")
    
    return results

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python classification_multicase.py [dataset_name]")
        sys.exit(1)
    
    run_analysis()