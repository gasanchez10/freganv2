import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Target: Get baseline around 94-95% so CATE stays below it
N = 1000
mu = 0
sigma = 3
np.random.seed(123)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def round_list_items(lst):
    return [round(item) for item in lst]

# Generate data
Z1 = np.random.normal(mu, 1.5*sigma, N)
Z2 = np.random.normal(mu, sigma, N)
UT = np.random.normal(mu, sigma, N)
TT = sigmoid(Z1 + UT)
T = round_list_items(TT)
T = np.asarray(T).reshape([N,])

UJ = np.random.normal(mu, sigma, N)
UP = np.random.normal(mu, sigma, N)
UD = np.random.normal(mu, sigma, N)
U = np.random.normal(mu, sigma, N)

J = 3*T - 2*Z1 + 3*Z2 + UJ
P = J - 2*T + 2*Z2 + 0.5*Z1 + UP
D = 2*J + P + 2*T + Z1 + Z2 + UD

# Try much more extreme weights to create harder problem
extreme_weights = [
    # Very challenging - more extremes
    [15.0, 1.5, 1.2, 0.8, 0.6, 0.4, 0.2],
    [18.0, 1.8, 1.5, 1.0, 0.8, 0.5, 0.25],
    [20.0, 2.0, 1.8, 1.2, 1.0, 0.6, 0.3],
    [25.0, 2.5, 2.0, 1.5, 1.2, 0.8, 0.4],
    [12.0, 1.2, 1.0, 0.6, 0.5, 0.3, 0.15],
    [30.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
]

print("Testing extreme weights for ~94-95% baseline accuracy:")
print("="*60)

results = []

for i, weights in enumerate(extreme_weights):
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = weights
    
    # Generate Y_aux with current weights
    linear_combo = T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U
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
    
    # Train regression model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(train_x_with_t, train_y_continuous)
    
    # Predict and evaluate
    y_pred_continuous = rf.predict(test_x_with_t)
    y_pred_binary = (y_pred_continuous > 0.5).astype(int)
    
    mse = mean_squared_error(test_y_continuous, y_pred_continuous)
    accuracy = accuracy_score(test_y_binary, y_pred_binary)
    
    # Calculate distribution stats
    near_0 = np.sum(Y_aux < 0.01)  # Very close to 0
    near_1 = np.sum(Y_aux > 0.99)  # Very close to 1
    extreme_total = near_0 + near_1
    extreme_ratio = extreme_total / N
    
    results.append({
        'weights': weights,
        'accuracy': accuracy,
        'mse': mse,
        'extreme_ratio': extreme_ratio,
        'near_0': near_0,
        'near_1': near_1
    })
    
    print(f"\nWeight Set {i+1}: {weights}")
    print(f"  MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  Prob mean: {np.mean(Y_aux):.3f}, std: {np.std(Y_aux):.3f}")
    print(f"  Extremes (<0.01 or >0.99): {extreme_total} ({extreme_ratio:.1%})")
    print(f"  Linear range: [{np.min(linear_combo):.1f}, {np.max(linear_combo):.1f}]")

# Find best candidate for 94-95% range
print("\n" + "="*60)
print("ANALYSIS FOR 94-95% BASELINE TARGET:")
print("="*60)

target_range = (0.94, 0.95)
best_candidate = None

for i, result in enumerate(results):
    if target_range[0] <= result['accuracy'] <= target_range[1]:
        print(f"✓ Set {i+1}: Accuracy {result['accuracy']:.3f} - IN TARGET RANGE!")
        if best_candidate is None:
            best_candidate = (i, result)
    else:
        print(f"  Set {i+1}: Accuracy {result['accuracy']:.3f}")

if best_candidate is None:
    # Find closest to 94.5%
    target = 0.945
    best_idx = min(range(len(results)), key=lambda i: abs(results[i]['accuracy'] - target))
    best_candidate = (best_idx, results[best_idx])
    print(f"\nClosest to target: Set {best_idx+1} with {results[best_idx]['accuracy']:.3f}")

if best_candidate:
    idx, result = best_candidate
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = result['weights']
    
    print(f"\n" + "="*60)
    print("RECOMMENDED UPDATE FOR LOWER BASELINE:")
    print("="*60)
    print(f"Expected baseline: ~{result['accuracy']:.1%}")
    print(f"Extreme values: {result['extreme_ratio']:.1%}")
    print(f"\nReplace in synthetic_class_data.py:")
    print(f"Y_aux = sigmoid({T_coef}*T + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D + {J_coef}*J + {P_coef}*P + {U_coef}*U)")
    print(f"Y_cf_Manual_aux = sigmoid({T_coef}*T_CF + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D_CF + {J_coef}*J_CF + {P_coef}*P_CF + {U_coef}*U)")
    print(f"\nThis should ensure CATE accuracy stays BELOW {result['accuracy']:.1%} baseline across all alphas!")
else:
    print("No suitable weights found. Try even more extreme values.")