import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# Configuration parameters
N = 1000
mu = 0
sigma = 3
np.random.seed(123)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def round_list_items(lst):
    return [round(item) for item in lst]

# Generate data (same as original)
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

# Test different weight combinations to find challenging but not extreme distributions
weight_sets = [
    # Try to find sweet spot: challenging but not too extreme
    [1.0, 0.15, 0.12, 0.08, 0.06, 0.04, 0.02],   # Moderate challenge
    [1.5, 0.2, 0.15, 0.1, 0.08, 0.06, 0.03],     # Bit more challenging
    [2.0, 0.25, 0.2, 0.12, 0.1, 0.08, 0.04],     # More challenging
    [0.8, 0.12, 0.1, 0.06, 0.05, 0.03, 0.015],   # Less challenging
    [1.2, 0.18, 0.14, 0.09, 0.07, 0.05, 0.025],  # Medium challenge
    [0.6, 0.1, 0.08, 0.05, 0.04, 0.025, 0.01],   # Easier
]

results = []

for i, weights in enumerate(weight_sets):
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
    
    # Train regression model (like in classification-multicase-reg)
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
        'mse': mse,
        'accuracy': accuracy,
        'mean_prob': np.mean(Y_aux),
        'std_prob': np.std(Y_aux),
        'balance_score': balance_score,
        'near_0': near_0,
        'near_1': near_1,
        'middle': middle,
        'linear_range': [np.min(linear_combo), np.max(linear_combo)]
    })
    
    print(f"\nWeight Set {i+1}: {weights}")
    print(f"  MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")
    print(f"  Prob mean: {np.mean(Y_aux):.3f}, std: {np.std(Y_aux):.3f}")
    print(f"  Distribution: <0.1: {near_0}, 0.1-0.9: {middle}, >0.9: {near_1}")
    print(f"  Balance score: {balance_score:.3f}")
    print(f"  Linear range: [{np.min(linear_combo):.2f}, {np.max(linear_combo):.2f}]")

# Find best candidate (accuracy around 0.94-0.96, not too extreme)
print("\n" + "="*60)
print("ANALYSIS FOR CHALLENGING BUT REASONABLE DISTRIBUTION:")
print("="*60)

target_accuracy_range = (0.94, 0.96)
good_candidates = []

for i, result in enumerate(results):
    if target_accuracy_range[0] <= result['accuracy'] <= target_accuracy_range[1]:
        if result['balance_score'] > 0.3:  # Not too extreme
            good_candidates.append((i, result))

if good_candidates:
    print("Good candidates (accuracy 94-96%, reasonable balance):")
    for i, result in good_candidates:
        print(f"  Set {i+1}: Accuracy {result['accuracy']:.3f}, Balance {result['balance_score']:.3f}")
        print(f"    Weights: {result['weights']}")
else:
    print("No perfect candidates found. Best options:")
    # Sort by how close to target accuracy range
    sorted_results = sorted(enumerate(results), 
                           key=lambda x: min(abs(x[1]['accuracy'] - target_accuracy_range[0]),
                                            abs(x[1]['accuracy'] - target_accuracy_range[1])))
    
    for i in range(min(3, len(sorted_results))):
        idx, result = sorted_results[i]
        print(f"  Set {idx+1}: Accuracy {result['accuracy']:.3f}, Balance {result['balance_score']:.3f}")
        print(f"    Weights: {result['weights']}")

print("\nRECOMMENDATION:")
print("Choose weights that give accuracy around 94-96% with reasonable probability distribution")
print("This will ensure fairness regularization stays below baseline but isn't too extreme.")