import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Change this variable to switch datasets
DATASET = "synthetic_class_data"  # Options: "synthetic_class_data", "german_credit"

def set_global_determinism(seed=42):
    """Set global determinism for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()

def get_optimal_nn_params(dataset_name):
    """Get optimal neural network parameters for dataset"""
    if dataset_name == "synthetic_class_data":
        return {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.3,
            'l2_reg': 0.01,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }
    elif dataset_name == "german_credit":
        return {
            'hidden_layers': [32, 16],
            'dropout_rate': 0.2,
            'l2_reg': 0.005,
            'learning_rate': 0.0005,
            'epochs': 150,
            'batch_size': 16
        }
    else:
        # Default parameters
        return {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.3,
            'l2_reg': 0.01,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }

def create_neural_network(input_dim, dataset_name, random_seed=42):
    """Create neural network with dataset-specific optimization"""
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.keras.backend.clear_session()
    
    params = get_optimal_nn_params(dataset_name)
    
    model = keras.Sequential()
    
    # Input layer + first hidden layer
    model.add(layers.Dense(
        params['hidden_layers'][0], 
        activation='relu', 
        input_shape=(input_dim,),
        kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.l2(params['l2_reg'])
    ))
    model.add(layers.Dropout(params['dropout_rate'], seed=random_seed))
    
    # Additional hidden layers
    for hidden_size in params['hidden_layers'][1:]:
        model.add(layers.Dense(
            hidden_size,
            activation='relu',
            kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
            bias_initializer=keras.initializers.Zeros(),
            kernel_regularizer=keras.regularizers.l2(params['l2_reg'])
        ))
        model.add(layers.Dropout(params['dropout_rate'], seed=random_seed))
    
    # Output layer
    model.add(layers.Dense(
        1, 
        activation='sigmoid',
        kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
        bias_initializer=keras.initializers.Zeros()
    ))
    
    return model

def combined_bce_loss(alpha, probunos, probceros):
    """Combined binary cross-entropy loss function with probability constants"""
    def loss_fn(y_true, y_pred):
        # Ensure proper shapes
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_factual = y_true[:, 0:1]
        y_counterfactual = y_true[:, 1:2]
        
        # Standard BCE for factual
        bce_factual = tf.keras.losses.binary_crossentropy(y_factual, y_pred)
        
        # BCE for counterfactual with averaging
        average = (y_counterfactual * probceros + y_factual * probunos)
        bce_counterfactual = tf.keras.losses.binary_crossentropy(tf.round(average), y_pred)
        
        # Ensure same shape for addition
        bce_factual = tf.reduce_mean(bce_factual)
        bce_counterfactual = tf.reduce_mean(bce_counterfactual)
        
        # Combined loss
        return alpha * bce_factual + (1 - alpha) * bce_counterfactual
    return loss_fn

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, cf_type="", dataset_name=""):
    """Calculate accuracy using neural network with optimized parameters"""
    alpha_range = np.arange(0, 1.1, 0.1)
    accuracy_results = {}
    
    # Get optimal parameters for this dataset
    params = get_optimal_nn_params(dataset_name)
    print(f"Optimizing NN parameters for {cf_type}...")
    print(f"Optimal params: {params}")
    
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
    
    for a_f in alpha_range:
        # Reset seeds for reproducibility - use constant seed for baseline models
        if cf_type == "Baseline" or cf_type == "PureBaseline":
            set_global_determinism(seed=42)  # Constant seed for baseline
        else:
            set_global_determinism(seed=42 + int(a_f * 10))  # Variable seed for CATE
        
        # Create model with dataset-specific parameters - use constant seed for baseline
        if cf_type == "Baseline" or cf_type == "PureBaseline":
            model = create_neural_network(train_x.shape[1], dataset_name, random_seed=42)
        else:
            model = create_neural_network(train_x.shape[1], dataset_name, random_seed=42 + int(a_f * 10))
        
        # Prepare training data with both factual and counterfactual labels
        y_combined = np.column_stack([train_y_factual.values, train_y_counterfactual.values])
        
        # Compile with optimized parameters
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=params['learning_rate'], 
                beta_1=0.9, 
                beta_2=0.999
            ),
            loss=combined_bce_loss(a_f, probunos, probceros),
            metrics=['accuracy']
        )
        
        # Train model with dataset-specific parameters
        model.fit(
            train_x.values, y_combined,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            verbose=0,
            shuffle=True,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Predict on test set
        test_pred_prob = model.predict(test_x.values, verbose=0)
        test_pred = (test_pred_prob > 0.5).astype(int).flatten()
        
        # Save predictions for this alpha in dataset folder
        dataset_folder = f"./{dataset_name}_predictions"
        pred_folder = f"{dataset_folder}/{cf_type}_predictions_alpha_{a_f:.1f}"
        os.makedirs(pred_folder, exist_ok=True)
        
        # Save predictions and training data
        pd.DataFrame(test_pred, columns=['predictions']).to_csv(f"{pred_folder}/test_predictions.csv", index=False)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_y.values, test_pred)
        accuracy_results[a_f] = accuracy
        
        print(f"Alpha: {a_f:.1f}, Accuracy: {accuracy:.4f}")
    
    return accuracy_results

def load_cate_counterfactuals(dataset):
    """Load CATE counterfactuals for specified dataset from local results"""
    train_potential = pd.read_csv(f"./{dataset}_cate_results/train_potential_y_1.0.0_binary_cate.csv")
    train_t = pd.read_csv(f"./{dataset}_cate_results/train_t_1.0.0_binary_cate.csv")
    
    # Extract counterfactuals based on treatment assignment
    train_cf_cate = []
    for i in range(len(train_potential)):
        if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
            train_cf_cate.append(train_potential.iloc[i, 1])
        else:  # Treatment group, counterfactual is Y0
            train_cf_cate.append(train_potential.iloc[i, 0])
    
    return pd.DataFrame(train_cf_cate, columns=['0'])

def load_cate_counterfactuals_german_credit(dataset):
    """Load CATE counterfactuals for German credit (continuous naming)"""
    train_potential = pd.read_csv(f"./{dataset}_cate_results/train_potential_y_1.0.0_continuous_cate.csv")
    train_t = pd.read_csv(f"./{dataset}_cate_results/train_t_1.0.0_continuous_cate.csv")
    
    # Extract counterfactuals based on treatment assignment
    train_cf_cate = []
    for i in range(len(train_potential)):
        if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
            train_cf_cate.append(train_potential.iloc[i, 1])
        else:  # Treatment group, counterfactual is Y0
            train_cf_cate.append(train_potential.iloc[i, 0])
    
    return pd.DataFrame(train_cf_cate, columns=['0'])

def run_fair_model_analysis(dataset):
    """Run fair model analysis comparing baseline vs CATE counterfactuals"""
    # Set global determinism
    set_global_determinism(seed=42)
    
    # Load original data for proper comparison
    if dataset == "synthetic_class_data":
        version = "1.0.0"
        orig_path = f"../{dataset}/data/"
        
        train_x = pd.read_csv(f"{orig_path}train_x_{version}_binary2025.csv")
        train_y = pd.read_csv(f"{orig_path}train_y_{version}_binary2025.csv")
        train_t = pd.read_csv(f"{orig_path}train_t_{version}_binary2025.csv")
        test_x = pd.read_csv(f"{orig_path}test_x_{version}_binary2025.csv")
        test_y = pd.read_csv(f"{orig_path}test_y_{version}_binary2025.csv")
        test_t = pd.read_csv(f"{orig_path}test_t_{version}_binary2025.csv")
        
        # Load CATE counterfactuals
        train_cf_cate = load_cate_counterfactuals(dataset)
        
    elif dataset == "german_credit":
        version = "1.0.0"
        orig_path = f"../{dataset}/data/"
        
        train_x = pd.read_csv(f"{orig_path}train_x_{version}_continuous.csv")
        train_y = pd.read_csv(f"{orig_path}train_y_{version}_continuous.csv")
        train_t = pd.read_csv(f"{orig_path}train_t_{version}_continuous.csv")
        test_x = pd.read_csv(f"{orig_path}test_x_{version}_continuous.csv")
        test_y = pd.read_csv(f"{orig_path}test_y_{version}_continuous.csv")
        test_t = pd.read_csv(f"{orig_path}test_t_{version}_continuous.csv")
        
        # Load CATE counterfactuals
        train_cf_cate = load_cate_counterfactuals_german_credit(dataset)
    
    # Create baseline counterfactuals (counterfactual = factual)
    train_cf_baseline = train_y.copy()
    
    # Create pure baseline where counterfactual=factual for fairness analysis
    train_cf_pure_baseline = train_y.copy()
    
    datasets = {
        "Baseline": train_cf_baseline,
        "CATE": train_cf_cate,
        "PureBaseline": train_cf_pure_baseline
    }
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    print(f"=== {dataset.upper().replace('_', ' ')} FAIR MODEL ANALYSIS ===")
    
    all_results = {}
    
    for i, (cf_type, cf_data) in enumerate(datasets.items()):
        print(f"\n=== {cf_type} Counterfactuals ===")
        accuracy_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, cf_type, dataset)
        all_results[cf_type] = accuracy_results
        
        alphas = list(accuracy_results.keys())
        plt.plot(alphas, list(accuracy_results.values()), 
                color=colors[i], marker=markers[i], 
                linestyle='-', label=cf_type, linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{dataset.replace("_", " ").title()} - Baseline vs CATE Counterfactuals', fontsize=14)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{dataset}_fair_model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save results
    results_data = {'alpha': list(all_results['Baseline'].keys())}
    for cf_type, accuracy_results in all_results.items():
        results_data[f'accuracy_{cf_type.lower()}'] = list(accuracy_results.values())
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{dataset}_fair_model_results.csv', index=False)
    
    # Find optimal results
    print("\n=== OPTIMAL RESULTS ===")
    for cf_type, accuracy_results in all_results.items():
        optimal_alpha = max(accuracy_results, key=accuracy_results.get)
        print(f"{cf_type} - Alpha: {optimal_alpha:.1f}, Accuracy: {accuracy_results[optimal_alpha]:.4f}")

if __name__ == "__main__":
    # Check if dataset is specified as command line argument
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    
    # Validate dataset choice
    valid_datasets = ["synthetic_class_data", "german_credit"]
    if DATASET not in valid_datasets:
        print(f"Error: Dataset must be one of {valid_datasets}")
        print(f"Usage: python fair_model.py [dataset_name]")
        sys.exit(1)
    
    run_fair_model_analysis(DATASET)