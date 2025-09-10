import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Let's try even more challenging weights to keep CATE below baseline
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

# Try more challenging weights - higher coefficients to create more extremes
challenging_weights = [
    # More extreme to make problem harder
    [8.0, 0.8, 0.7, 0.4, 0.35, 0.2, 0.1],
    [10.0, 1.0, 0.9, 0.5, 0.4, 0.25, 0.12],
    [12.0, 1.2, 1.0, 0.6, 0.5, 0.3, 0.15],
    [7.0, 0.7, 0.6, 0.35, 0.3, 0.18, 0.09],
    [9.0, 0.9, 0.8, 0.45, 0.38, 0.22, 0.11],
    [6.0, 0.6, 0.5, 0.3, 0.25, 0.15, 0.08],
]

print("Testing more challenging weights to keep CATE below baseline:")
print("="*60)

results = []

for i, weights in enumerate(challenging_weights):
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
    near_0 = np.sum(Y_aux < 0.1)
    near_1 = np.sum(Y_aux > 0.9)
    middle = np.sum((Y_aux >= 0.1) & (Y_aux <= 0.9))
    balance_score = middle / N
    
    results.append({
        'weights': weights,
        'accuracy': accuracy,
        'mse': mse,
        'balance_score': balance_score,
        'near_0': near_0,
        'near_1': near_1,
        'extremes_ratio': (near_0 + near_1) / N
    })
    
    print(f"\nWeight Set {i+1}: {weights}")
    print(f"  MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  Prob mean: {np.mean(Y_aux):.3f}, std: {np.std(Y_aux):.3f}")
    print(f"  Distribution: <0.1: {near_0}, 0.1-0.9: {middle}, >0.9: {near_1}")
    print(f"  Balance score: {balance_score:.3f}, Extremes ratio: {(near_0 + near_1) / N:.3f}")
    print(f"  Linear range: [{np.min(linear_combo):.2f}, {np.max(linear_combo):.2f}]")

# Find weights that give lower baseline accuracy (around 96-97%)
print("\n" + "="*60)
print("BEST CANDIDATES FOR LOWER BASELINE (96-97% range):")
print("="*60)

target_range = (0.96, 0.97)
good_candidates = []

for i, result in enumerate(results):
    if target_range[0] <= result['accuracy'] <= target_range[1]:
        good_candidates.append((i, result))

if good_candidates:
    print("Candidates in 96-97% accuracy range:")
    for i, result in good_candidates:
        print(f"  Set {i+1}: Accuracy {result['accuracy']:.3f}, Extremes {result['extremes_ratio']:.3f}")
        print(f"    Weights: {result['weights']}")
else:
    # Find closest to 96.5%
    target = 0.965
    sorted_results = sorted(enumerate(results), key=lambda x: abs(x[1]['accuracy'] - target))
    
    print("Closest to 96.5% baseline:")
    for i in range(min(3, len(sorted_results))):
        idx, result = sorted_results[i]
        print(f"  Set {idx+1}: Accuracy {result['accuracy']:.3f} (diff: {abs(result['accuracy'] - target)*100:.1f}%)")
        print(f"    Weights: {result['weights']}")
        print(f"    Extremes ratio: {result['extremes_ratio']:.3f}")

if results:
    # Pick the one closest to 96.5%
    target = 0.965
    best_idx = min(range(len(results)), key=lambda i: abs(results[i]['accuracy'] - target))
    best_result = results[best_idx]
    
    print(f"\n" + "="*60)
    print("RECOMMENDED WEIGHTS FOR LOWER BASELINE:")
    print("="*60)
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = best_result['weights']
    print(f"Set {best_idx+1}: Expected baseline ~{best_result['accuracy']:.1%}")
    print(f"Y_aux = sigmoid({T_coef}*T + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D + {J_coef}*J + {P_coef}*P + {U_coef}*U)")
    print(f"Y_cf_Manual_aux = sigmoid({T_coef}*T_CF + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D_CF + {J_coef}*J_CF + {P_coef}*P_CF + {U_coef}*U)")
    print(f"\nThis should keep CATE performance below {best_result['accuracy']:.1%} baseline!")