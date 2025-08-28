import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

def set_global_determinism(seed=42):
    """
    Set global determinism for complete reproducibility across all libraries.
    This ensures identical results every time the code is run.
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    print(f"Global determinism set with seed: {seed}")

def create_neural_network(input_dim, random_seed=42):
    """Create neural network with deterministic weight initialization"""
    # Set all random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Clear any existing models
    tf.keras.backend.clear_session()
    
    # Create model with fixed weight initialization
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,),
                    kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros(),
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3, seed=random_seed),
        layers.Dense(32, activation='relu',
                    kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros(),
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3, seed=random_seed),
        layers.Dense(1, activation='sigmoid',
                    kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros())
    ])
    return model

def combined_bce_loss(alpha, probunos, probceros):
    """Combined binary cross-entropy loss function with probability constants"""
    def loss_fn(y_true, y_pred):
        # y_true should contain both factual and counterfactual labels
        y_factual = y_true[:, 0:1]
        y_counterfactual = y_true[:, 1:2]
        
        # Standard BCE for factual
        bce_factual = tf.keras.losses.binary_crossentropy(y_factual, y_pred)
        
        # BCE for counterfactual with averaging
        average=(y_counterfactual * probceros + y_factual * probunos)
        bce_counterfactual = tf.keras.losses.binary_crossentropy(tf.round(average), y_pred)
        
        # Combined loss
        return alpha * bce_factual + (1 - alpha) * bce_counterfactual
    return loss_fn

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, combination_type="average", dataset_name="", alpha_range=None):
    """Calculate accuracy using neural network with combined BCE loss"""
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
        print(f"  Training neural network for alpha {a_f:.1f}")
        
        # Reset seeds before each model creation for consistency
        set_global_determinism(seed=42 + int(a_f * 10))  # Unique but deterministic seed per alpha
        
        # Create model with fixed seed for reproducibility
        model = create_neural_network(train_x.shape[1], random_seed=42 + int(a_f * 10))
        
        # Prepare training data with both factual and counterfactual labels
        y_combined = np.column_stack([train_y_factual.values, train_y_counterfactual.values])
        
        # Compile with combined loss and fixed optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss=combined_bce_loss(a_f, probunos, probceros),
            metrics=['accuracy']
        )
        
        # Train model with fixed parameters for reproducibility
        model.fit(
            train_x.values, y_combined,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            shuffle=True,  # Shuffle with fixed seed
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Predict on test set
        test_pred_prob = model.predict(test_x.values, verbose=0)
        test_pred = (test_pred_prob > 0.5).astype(int).flatten()
        
        # Predict on train set
        train_pred_prob = model.predict(train_x.values, verbose=0)
        train_pred = (train_pred_prob > 0.5).astype(int).flatten()
        
        # Create directory for this specific case
        case_dir = f"./predictions/{dataset_name.lower().replace(' ', '_')}_{combination_type}_alpha_{a_f:.1f}"
        os.makedirs(case_dir, exist_ok=True)
        
        # Save all required datasets
        train_x.to_csv(f"{case_dir}/train_x_1.0.0_binary2025.csv", index=False)
        train_t.to_csv(f"{case_dir}/train_t_1.0.0_binary2025.csv", index=False)
        pd.DataFrame(train_pred, columns=['0']).to_csv(f"{case_dir}/train_y_1.0.0_binary2025.csv", index=False)
        test_x.to_csv(f"{case_dir}/test_x_1.0.0_binary2025.csv", index=False)
        test_t.to_csv(f"{case_dir}/test_t_1.0.0_binary2025.csv", index=False)
        pd.DataFrame(test_pred, columns=['0']).to_csv(f"{case_dir}/test_y_1.0.0_binary2025.csv", index=False)
        
        # Save predictions comparison
        combined_actual = np.concatenate([train_y_factual.values.ravel(), test_y.values.ravel()])
        combined_predicted = np.concatenate([train_pred, test_pred])
        predictions_df = pd.DataFrame({
            'actual': combined_actual,
            'predicted': combined_predicted
        })
        predictions_df.to_csv(f"{case_dir}/predictions.csv", index=False)
        
        print(f"  Saved {len(predictions_df)} predictions ({len(train_pred)} train + {len(test_pred)} test)")
        
        # Calculate test accuracy using standard BCE
        test_loss = tf.keras.losses.binary_crossentropy(test_y.values, test_pred_prob).numpy().mean()
        accuracy = accuracy_score(test_y.values, test_pred)
        accuracy_results[a_f] = accuracy
        
        print(f"Alpha: {a_f:.1f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return accuracy_results

def load_ganite_counterfactuals(alpha_ganite):
    """Load GANITE counterfactuals for a specific alpha"""
    base_path_ganite = "./counterfactuals/"
    test_y_hat = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/test_y_hat_1.0.0_binary_ganite_alpha{alpha_ganite}.csv")
    train_t = pd.read_csv(f"{base_path_ganite}alpha_{alpha_ganite}/train_t_1.0.0_binary_ganite_alpha{alpha_ganite}.csv")
    
    # Get train portion from test_y_hat
    train_size = len(train_t)
    train_y_hat = test_y_hat.iloc[:train_size]
    
    # Extract counterfactuals
    train_cf_ganite = []
    for index, row in train_y_hat.iterrows():
        cf_treatment = 1 - int(train_t.iloc[index, 0])
        train_cf_ganite.append(row[str(cf_treatment)])
    
    return pd.DataFrame(train_cf_ganite, columns=['0'])

def load_cate_counterfactuals():
    """Load CATE counterfactuals"""
    try:
        train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_binary_cate.csv")
        train_t = pd.read_csv("./data/train_t_1.0.0_binary2025.csv")
        
        # Extract counterfactuals based on treatment assignment
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

def run_fair_model_analysis():
    """Run fair model analysis with all counterfactual datasets"""
    # Ensure complete reproducibility
    set_global_determinism(seed=42)
    # Load synthetic data
    version = "1.0.0"
    base_path = "./data/"
    
    # Load required datasets
    train_x = pd.read_csv(f"{base_path}train_x_{version}_binary2025.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_binary2025.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_binary2025.csv")
    train_cf_manual = pd.read_csv(f"{base_path}train_cf_manual_{version}_binary2025.csv")
    train_cf_random = pd.read_csv(f"{base_path}train_cf_random_{version}_binary2025.csv")
    train_cf_y = pd.read_csv(f"{base_path}train_cf_y_{version}_binary2025.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_binary2025.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_binary2025.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_binary2025.csv")
    
    # Define combination types (only average)
    combination_types = ["average"]
    counterfactual_datasets = {
        "Manual": train_cf_manual,
        "Random": train_cf_random,
        "True": train_cf_y
    }
    
    # Load GANITE datasets
    ganite_alphas = [5, 6, 7, 8]
    ganite_datasets = {}
    for ganite_alpha in ganite_alphas:
        try:
            ganite_datasets[f"GANITE_α{ganite_alpha}"] = load_ganite_counterfactuals(ganite_alpha)
        except Exception as e:
            print(f"Error loading GANITE alpha {ganite_alpha}: {e}")
    
    # Load CATE counterfactuals
    cate_datasets = {}
    cate_cf = load_cate_counterfactuals()
    if cate_cf is not None:
        cate_datasets["CATE"] = cate_cf
    
    # Combine all datasets
    all_datasets = {**counterfactual_datasets, **ganite_datasets, **cate_datasets}
    
    # Create plots for each combination type
    for combo_type in combination_types:
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h']
        
        print(f"\n{'='*50}")
        print(f"COMBINATION TYPE: {combo_type.upper()}")
        print(f"{'='*50}")
        
        all_results = {}
        
        for i, (dataset_name, cf_data) in enumerate(all_datasets.items()):
            print(f"\n=== {dataset_name} Counterfactuals ({combo_type}) ===")
            accuracy_results = fair_metrics_calculator(train_x, train_y, cf_data, test_x, test_y, train_t, test_t, combination_type=combo_type, dataset_name=dataset_name)
            all_results[dataset_name] = accuracy_results
            
            alphas = list(accuracy_results.keys())
            plt.plot(alphas, list(accuracy_results.values()), 
                    color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                    linestyle='-', label=dataset_name, linewidth=2, markersize=8)
        
        plt.xlabel('Alpha (Linear Combination Factor)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Fair Model Performance - {combo_type.capitalize()} Combination', fontsize=14)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"fair_model_{combo_type}_combination.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"fair_model_{combo_type}_results.csv")
        
        print(f"\nResults saved to fair_model_{combo_type}_results.csv")
        print(f"Plot saved to fair_model_{combo_type}_combination.png")

if __name__ == "__main__":
    run_fair_model_analysis()