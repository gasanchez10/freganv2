import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Configuration parameters
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

# Try much smaller weights to make the problem harder (closer to random)
harder_weight_sets = [
    # Very small weights - make it harder to predict
    [0.1, 0.02, 0.015, 0.01, 0.008, 0.005, 0.002],
    [0.15, 0.025, 0.02, 0.012, 0.01, 0.007, 0.003],
    [0.08, 0.015, 0.012, 0.008, 0.006, 0.004, 0.001],
    [0.12, 0.02, 0.018, 0.01, 0.008, 0.006, 0.002],
    # Add some noise/randomness
    [0.05, 0.01, 0.008, 0.005, 0.004, 0.002, 0.001],
    [0.25, 0.03, 0.025, 0.015, 0.012, 0.008, 0.004],
]

print("Testing harder weight combinations for lower baseline accuracy:")
print("="*60)

best_candidate = None
target_accuracy = 0.94  # Target around 94%

for i, weights in enumerate(harder_weight_sets):
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
    
    print(f"\nWeight Set {i+1}: {weights}")
    print(f"  MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  Prob mean: {np.mean(Y_aux):.3f}, std: {np.std(Y_aux):.3f}")
    print(f"  Distribution: <0.1: {near_0}, 0.1-0.9: {middle}, >0.9: {near_1}")
    print(f"  Balance score: {balance_score:.3f}")
    print(f"  Linear range: [{np.min(linear_combo):.2f}, {np.max(linear_combo):.2f}]")
    
    # Check if this is close to our target
    if abs(accuracy - target_accuracy) < 0.02 and balance_score > 0.4:
        if best_candidate is None or abs(accuracy - target_accuracy) < abs(best_candidate['accuracy'] - target_accuracy):
            best_candidate = {
                'weights': weights,
                'accuracy': accuracy,
                'mse': mse,
                'balance_score': balance_score,
                'set_num': i+1
            }

print("\n" + "="*60)
print("RECOMMENDATION FOR ~94% BASELINE ACCURACY:")
print("="*60)

if best_candidate:
    print(f"Best candidate: Set {best_candidate['set_num']}")
    print(f"  Weights: {best_candidate['weights']}")
    print(f"  Accuracy: {best_candidate['accuracy']:.4f}")
    print(f"  MSE: {best_candidate['mse']:.4f}")
    print(f"  Balance score: {best_candidate['balance_score']:.3f}")
    
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = best_candidate['weights']
    print(f"\nUpdate synthetic_class_data.py with:")
    print(f"Y_aux = sigmoid({T_coef}*T + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D + {J_coef}*J + {P_coef}*P + {U_coef}*U)")
else:
    print("No ideal candidate found. Try even smaller weights or add noise to make problem harder.")
    print("Current weights might be making the problem too easy to solve.")