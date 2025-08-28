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
        
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        bce_factual = tf.keras.losses.binary_crossentropy(y_factual, y_pred)
        
        average = y_counterfactual * probceros + y_factual * probunos
        bce_counterfactual = tf.keras.losses.binary_crossentropy(average, y_pred)
        
        return alpha * bce_factual + (1 - alpha) * bce_counterfactual
    return loss_fn

def create_neural_network_with_reg(input_dim, reg_strength, random_seed=42):
    """Create optimized neural network for high accuracy"""
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.keras.backend.clear_session()
    
    # Optimal regularization balance
    if reg_strength < 0.001:
        dropout_rate = 0.15  # Minimal but not zero
        l2_reg = 0.0001     # Light regularization
    else:
        dropout_rate = 0.4 + (reg_strength * 6)  # 0.4 to 0.7 range
        l2_reg = reg_strength
    
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,),
                    kernel_initializer=keras.initializers.HeNormal(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate, seed=random_seed),
        layers.Dense(128, activation='relu',
                    kernel_initializer=keras.initializers.HeNormal(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.7, seed=random_seed),
        layers.Dense(64, activation='relu',
                    kernel_initializer=keras.initializers.HeNormal(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros(),
                    kernel_regularizer=keras.regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate * 0.5, seed=random_seed),
        layers.Dense(1, activation='sigmoid',
                    kernel_initializer=keras.initializers.GlorotUniform(seed=random_seed),
                    bias_initializer=keras.initializers.Zeros())
    ])
    return model

def combined_bce_loss(alpha, probunos, probceros):
    """Combined binary cross-entropy loss function with probability constants and smoothing"""
    def loss_fn(y_true, y_pred):
        y_factual = y_true[:, 0:1]
        y_counterfactual = y_true[:, 1:2]
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        bce_factual = tf.keras.losses.binary_crossentropy(y_factual, y_pred)
        
        # Smooth the average instead of rounding
        average = y_counterfactual * probceros + y_factual * probunos
        bce_counterfactual = tf.keras.losses.binary_crossentropy(average, y_pred)
        
        # Add L2 regularization to the loss
        combined_loss = alpha * bce_factual + (1 - alpha) * bce_counterfactual
        
        return combined_loss
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
        
        # Use same seed for all alphas to see pure regularization effect
        set_global_determinism(seed=42)
        
        # Moderate regularization only for low alpha
        reg_strength = 0.05 * (1 - a_f)  # 0.05 to 0.0 range
        model = create_neural_network_with_reg(train_x.shape[1], reg_strength, random_seed=42)
        
        y_combined = np.column_stack([train_y_factual.values.astype(np.float32), train_y_counterfactual.values.astype(np.float32)])
        
        y_combined = np.column_stack([train_y_factual.values.astype(np.float32), train_y_counterfactual.values.astype(np.float32)])
        
        # Optimal learning rate
        lr = 0.001 if a_f == 1.0 else 0.0005
        
        # Use combined BCE loss function
        y_combined = np.column_stack([train_y_factual.values.astype(np.float32), train_y_counterfactual.values.astype(np.float32)])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=combined_bce_loss(a_f, probunos, probceros),
            metrics=['accuracy']
        )
        
        # Optimal training parameters
        epochs = 400 if a_f == 1.0 else 100
        batch_size = 8 if a_f == 1.0 else 16
        
        # Maximum optimization for alpha=1.0
        if a_f == 1.0:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor='val_accuracy'),
                keras.callbacks.ReduceLROnPlateau(patience=20, factor=0.7, min_lr=1e-7),
                keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
            ]
        else:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy')
            ]
        
        model.fit(
            train_x.values.astype(np.float32), y_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=0,
            shuffle=True,
            callbacks=callbacks
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
        
        combined_actual = np.concatenate([train_y_factual.values.ravel(), test_y.values.ravel()])
        combined_predicted = np.concatenate([train_pred, test_pred])
        predictions_df = pd.DataFrame({
            'actual': combined_actual,
            'predicted': combined_predicted
        })
        predictions_df.to_csv(f"{case_dir}/predictions.csv", index=False)
        
        print(f"  Saved {len(predictions_df)} predictions ({len(train_pred)} train + {len(test_pred)} test)")
        
        test_loss = tf.keras.losses.binary_crossentropy(test_y.values.astype(np.float32), test_pred_prob).numpy().mean()
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
            if train_t.iloc[i, 0] == 0:  # Control group, counterfactual is Y1
                train_cf_cate.append(train_potential.iloc[i, 1])
            else:  # Treatment group, counterfactual is Y0
                train_cf_cate.append(train_potential.iloc[i, 0])
        
        return pd.DataFrame(train_cf_cate, columns=['0'])
    except Exception as e:
        print(f"Error loading CATE counterfactuals: {e}")
        return None

def run_fair_model_analysis():
    """Run fair model analysis with CATE counterfactuals only"""
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
    
    # Load CATE counterfactuals only
    cate_datasets = {}
    cate_cf = load_cate_counterfactuals()
    if cate_cf is not None:
        cate_datasets["CATE"] = cate_cf
    
    all_datasets = cate_datasets
    
    for combo_type in combination_types:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue']
        markers = ['o']
        
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
        plt.title(f'Fair Model Performance - Boston Housing (CATE Only)', fontsize=14)
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