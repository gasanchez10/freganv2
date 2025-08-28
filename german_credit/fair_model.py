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
    """Set global determinism for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    print(f"Global determinism set with seed: {seed}")

def combined_bce_loss(alpha, probunos, probceros):
    """Combined binary cross-entropy loss function with probability constants"""
    def loss_fn(y_true, y_pred):
        y_factual = y_true[:, 0:1]
        y_counterfactual = y_true[:, 1:2]
        
        bce_factual = tf.keras.losses.binary_crossentropy(y_factual, y_pred)
        
        if alpha >= 1.0:
            # Pure factual loss
            return bce_factual
        else:
            average = (y_counterfactual * probceros + y_factual * probunos)
            bce_counterfactual = tf.keras.losses.binary_crossentropy(tf.round(average), y_pred)
            return alpha * bce_factual + (1 - alpha) * bce_counterfactual
    return loss_fn

def create_neural_network(input_dim, random_seed=42):
    """Create neural network with flexible architecture"""
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.keras.backend.clear_session()
    
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(input_dim,),
                    kernel_initializer=keras.initializers.HeNormal(seed=random_seed),
                    bias_initializer=keras.initializers.Constant(0.1),
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2, seed=random_seed),
        layers.Dense(16, activation='relu',
                    kernel_initializer=keras.initializers.HeNormal(seed=random_seed),
                    bias_initializer=keras.initializers.Constant(0.1),
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.1, seed=random_seed),
        layers.Dense(1, activation='sigmoid',
                    kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
                    bias_initializer=keras.initializers.Constant(-0.5))
    ])
    return model

def fair_metrics_calculator(train_x, train_y_factual, train_y_counterfactual, test_x, test_y, train_t, test_t, combination_type="average", dataset_name="", alpha_range=None):
    """Calculate accuracy using neural network with combined BCE loss"""
    if alpha_range is None:
        alpha_range = np.arange(0, 1.1, 0.1)
    
    # Calculate probability constants
    probunos = train_t.iloc[:, 0].sum() / len(train_t)
    probceros = (len(train_t) - train_t.iloc[:, 0].sum()) / len(train_t)
    
    accuracy_results = {}
    
    for a_f in alpha_range:
        print(f"  Training neural network for alpha {a_f:.1f}")
        
        # For Baseline, always use same seed and alpha=1.0
        if dataset_name == "Baseline":
            set_global_determinism(seed=42)
            model = create_neural_network(train_x.shape[1], random_seed=42)
            actual_alpha = 1.0
        else:
            set_global_determinism(seed=42 + int(a_f * 10))
            model = create_neural_network(train_x.shape[1], random_seed=42 + int(a_f * 10))
            actual_alpha = a_f
        
        y_combined = np.column_stack([train_y_factual.values.astype(np.float32), train_y_counterfactual.values.astype(np.float32)])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss=combined_bce_loss(actual_alpha, probunos, probceros),
            metrics=['accuracy']
        )
        
        model.fit(
            train_x.values, y_combined,
            epochs=50,
            batch_size=16,
            validation_split=0.15,
            verbose=0,
            shuffle=True,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
            ]
        )
        
        test_pred_prob = model.predict(test_x.values.astype(np.float32), verbose=0)
        test_pred = (test_pred_prob > 0.5).astype(int).flatten()
        
        train_pred_prob = model.predict(train_x.values.astype(np.float32), verbose=0)
        train_pred = (train_pred_prob > 0.5).astype(int).flatten()
        
        case_dir = f"./predictions/{dataset_name.lower().replace(' ', '_')}_{combination_type}_alpha_{a_f:.1f}"
        os.makedirs(case_dir, exist_ok=True)
        
        train_x.to_csv(f"{case_dir}/train_x_1.0.0_binary.csv", index=False)
        train_t.to_csv(f"{case_dir}/train_t_1.0.0_binary.csv", index=False)
        pd.DataFrame(train_pred, columns=['0']).to_csv(f"{case_dir}/train_y_1.0.0_binary.csv", index=False)
        test_x.to_csv(f"{case_dir}/test_x_1.0.0_binary.csv", index=False)
        test_t.to_csv(f"{case_dir}/test_t_1.0.0_binary.csv", index=False)
        pd.DataFrame(test_pred, columns=['0']).to_csv(f"{case_dir}/test_y_1.0.0_binary.csv", index=False)
        
        print(f"  Saved {len(train_pred) + len(test_pred)} predictions ({len(train_pred)} train + {len(test_pred)} test)")
        
        test_loss = tf.keras.losses.binary_crossentropy(test_y.values, test_pred_prob).numpy().mean()
        
        combined_actual = np.concatenate([train_y_factual.values.ravel(), test_y.values.ravel()])
        combined_predicted = np.concatenate([train_pred, test_pred])
        predictions_df = pd.DataFrame({
            'actual': combined_actual,
            'predicted': combined_predicted
        })
        predictions_df.to_csv(f"{case_dir}/predictions.csv", index=False)
        
        accuracy = accuracy_score(test_y.values, test_pred)
        accuracy_results[a_f] = accuracy
        
        print(f"Alpha: {a_f:.1f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return accuracy_results

def load_cate_counterfactuals():
    """Load CATE counterfactuals"""
    try:
        train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_binary_cate.csv")
        train_t = pd.read_csv("./data/train_t_1.0.0_binary.csv")
        
        train_cf_cate = []
        for i in range(len(train_potential)):
            if train_t.iloc[i, 0] == 0:  # Control group (female), counterfactual is Y1 (male)
                train_cf_cate.append(train_potential.iloc[i, 1])
            else:  # Treatment group (male), counterfactual is Y0 (female)
                train_cf_cate.append(train_potential.iloc[i, 0])
        
        return pd.DataFrame(train_cf_cate, columns=['0'])
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return None

def run_fair_model_analysis():
    """Run fair model analysis with CATE counterfactuals"""
    set_global_determinism(seed=42)
    
    version = "1.0.0"
    base_path = "./data/"
    
    train_x = pd.read_csv(f"{base_path}train_x_{version}_binary.csv")
    train_y = pd.read_csv(f"{base_path}train_y_{version}_binary.csv")
    train_t = pd.read_csv(f"{base_path}train_t_{version}_binary.csv")
    test_x = pd.read_csv(f"{base_path}test_x_{version}_binary.csv")
    test_y = pd.read_csv(f"{base_path}test_y_{version}_binary.csv")
    test_t = pd.read_csv(f"{base_path}test_t_{version}_binary.csv")
    
    combination_types = ["average"]
    
    # Load CATE counterfactuals and add baseline
    cate_datasets = {}
    cate_cf = load_cate_counterfactuals()
    if cate_cf is not None:
        cate_datasets["CATE"] = cate_cf
    
    # Add baseline case where counterfactual = factual
    cate_datasets["Baseline"] = train_y
    
    all_datasets = cate_datasets
    
    for combo_type in combination_types:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red']
        markers = ['o', 's']
        
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
        plt.title(f'Fair Model Performance - German Credit', fontsize=14)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"fair_model_{combo_type}_combination.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"fair_model_{combo_type}_results.csv")
        
        print(f"\nResults saved to fair_model_{combo_type}_results.csv")
        print(f"Plot saved to fair_model_{combo_type}_combination.png")

if __name__ == "__main__":
    run_fair_model_analysis()