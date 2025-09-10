import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Add more noise to make the problem inherently harder
N = 1000
mu = 0
sigma = 3
np.random.seed(123)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def round_list_items(lst):
    return [round(item) for item in lst]

# Generate data with MORE NOISE
Z1 = np.random.normal(mu, 2.0*sigma, N)  # Increased noise
Z2 = np.random.normal(mu, 1.5*sigma, N)  # Increased noise
UT = np.random.normal(mu, 1.5*sigma, N)  # Increased noise
TT = sigmoid(Z1 + UT)
T = round_list_items(TT)
T = np.asarray(T).reshape([N,])

# More noise in error terms
UJ = np.random.normal(mu, 1.5*sigma, N)  # Increased
UP = np.random.normal(mu, 1.5*sigma, N)  # Increased
UD = np.random.normal(mu, 1.5*sigma, N)  # Increased
U = np.random.normal(mu, 2.0*sigma, N)   # Much more noise

J = 3*T - 2*Z1 + 3*Z2 + UJ
P = J - 2*T + 2*Z2 + 0.5*Z1 + UP
D = 2*J + P + 2*T + Z1 + Z2 + UD

# Test different approaches
approaches = [
    # Approach 1: Moderate weights with high noise
    {"weights": [5.0, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1], "extra_noise": 1.0},
    
    # Approach 2: Higher weights with extra noise
    {"weights": [8.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1], "extra_noise": 1.5},
    
    # Approach 3: Very high weights with lots of noise
    {"weights": [12.0, 1.0, 0.8, 0.5, 0.4, 0.25, 0.12], "extra_noise": 2.0},
    
    # Approach 4: Extreme weights with massive noise
    {"weights": [15.0, 1.2, 1.0, 0.6, 0.5, 0.3, 0.15], "extra_noise": 3.0},
]

print("Testing noisy approaches for lower baseline accuracy:")
print("="*60)

results = []

for i, approach in enumerate(approaches):
    weights = approach["weights"]
    extra_noise_std = approach["extra_noise"]
    
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = weights
    
    # Add extra noise to make problem harder
    extra_noise = np.random.normal(0, extra_noise_std, N)
    
    # Generate Y_aux with current weights + extra noise
    linear_combo = T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U + extra_noise
    Y_aux = sigmoid(linear_combo)
    Y = np.round(Y_aux).astype(int)
    
    # Create features and split data
    X = np.array([Z1, Z2, J, P, D]).T
    idx = np.random.permutation(N)
    train_idx = idx[:800]
    test_idx = idx[800:]
    
    train_x = X[train_idx]
    train_y_continuous = Y_aux[train_idx]
    train_y_binary = Y[train_idx]
    train_t = T[train_idx]
    
    test_x = X[test_idx]
    test_y_continuous = Y_aux[test_idx]
    test_y_binary = Y[test_idx]
    test_t = T[test_idx]
    
    # Add treatment to features
    train_x_with_t = np.column_stack([train_x, train_t])
    test_x_with_t = np.column_stack([test_x, test_t])
    
    # Train regression model with reduced complexity
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)  # Reduced complexity
    rf.fit(train_x_with_t, train_y_continuous)
    
    # Predict and evaluate
    y_pred_continuous = rf.predict(test_x_with_t)
    y_pred_binary = (y_pred_continuous > 0.5).astype(int)
    
    mse = mean_squared_error(test_y_continuous, y_pred_continuous)
    accuracy = accuracy_score(test_y_binary, y_pred_binary)
    
    results.append({
        'approach': i+1,
        'weights': weights,
        'extra_noise': extra_noise_std,
        'accuracy': accuracy,
        'mse': mse
    })
    
    print(f"\nApproach {i+1}:")
    print(f"  Weights: {weights}")
    print(f"  Extra noise std: {extra_noise_std}")
    print(f"  MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  Prob mean: {np.mean(Y_aux):.3f}, std: {np.std(Y_aux):.3f}")

# Find best approach for ~94% baseline
target = 0.94
best_approach = min(results, key=lambda x: abs(x['accuracy'] - target))

print(f"\n" + "="*60)
print("BEST APPROACH FOR ~94% BASELINE:")
print("="*60)
print(f"Approach {best_approach['approach']}: Accuracy {best_approach['accuracy']:.3f}")
print(f"Weights: {best_approach['weights']}")
print(f"Extra noise std: {best_approach['extra_noise']}")

T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = best_approach['weights']
extra_noise_std = best_approach['extra_noise']

print(f"\nUpdate synthetic_class_data.py with:")
print(f"# Add extra noise")
print(f"extra_noise = np.random.normal(0, {extra_noise_std}, N)")
print(f"Y_aux = sigmoid({T_coef}*T + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D + {J_coef}*J + {P_coef}*P + {U_coef}*U + extra_noise)")
print(f"Y_cf_Manual_aux = sigmoid({T_coef}*T_CF + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D_CF + {J_coef}*J_CF + {P_coef}*P_CF + {U_coef}*U + extra_noise)")
print(f"\nThis should keep CATE performance BELOW {best_approach['accuracy']:.1%} baseline!")