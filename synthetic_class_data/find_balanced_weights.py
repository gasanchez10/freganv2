import numpy as np
import matplotlib.pyplot as plt

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

# First, let's see the range of the linear combination before sigmoid
print("Variable ranges:")
print(f"T: {np.min(T):.2f} to {np.max(T):.2f}")
print(f"Z1: {np.min(Z1):.2f} to {np.max(Z1):.2f}")
print(f"Z2: {np.min(Z2):.2f} to {np.max(Z2):.2f}")
print(f"D: {np.min(D):.2f} to {np.max(D):.2f}")
print(f"J: {np.min(J):.2f} to {np.max(J):.2f}")
print(f"P: {np.min(P):.2f} to {np.max(P):.2f}")
print(f"U: {np.min(U):.2f} to {np.max(U):.2f}")

# Test very small weights to keep linear combination in reasonable range
# Target: keep linear combination roughly between -3 and +3 for balanced sigmoid
weight_sets = [
    # Very small weights
    [0.5, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01],
    [0.3, 0.03, 0.03, 0.01, 0.01, 0.01, 0.005],
    [0.2, 0.02, 0.02, 0.008, 0.008, 0.008, 0.003],
    [0.1, 0.01, 0.01, 0.004, 0.004, 0.004, 0.001],
    # Even smaller
    [0.05, 0.005, 0.005, 0.002, 0.002, 0.002, 0.0005],
    [0.02, 0.002, 0.002, 0.0008, 0.0008, 0.0008, 0.0002],
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

best_weights = None
best_balance = 0

for i, weights in enumerate(weight_sets):
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = weights
    
    # Calculate linear combination first
    linear_combo = T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U
    
    # Then apply sigmoid
    Y_aux = sigmoid(linear_combo)
    
    # Plot histogram
    axes[i].hist(Y_aux, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[i].set_xlabel('Y_aux Probabilities')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Set {i+1}: T={T_coef}, others scaled proportionally')
    axes[i].grid(True, alpha=0.3)
    
    # Calculate balance metrics
    mean_val = np.mean(Y_aux)
    std_val = np.std(Y_aux)
    near_0 = np.sum(Y_aux < 0.1)
    near_1 = np.sum(Y_aux > 0.9)
    middle = np.sum((Y_aux >= 0.1) & (Y_aux <= 0.9))
    balance_score = middle / N
    
    # Linear combination stats
    linear_min = np.min(linear_combo)
    linear_max = np.max(linear_combo)
    linear_mean = np.mean(linear_combo)
    
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nBalance: {balance_score:.3f}\nLinear: [{linear_min:.1f}, {linear_max:.1f}]'
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print(f"\nWeight set {i+1}:")
    print(f"  Weights: T={T_coef}, Z1={Z1_coef}, Z2={Z2_coef}, D={D_coef}, J={J_coef}, P={P_coef}, U={U_coef}")
    print(f"  Linear combo range: [{linear_min:.2f}, {linear_max:.2f}], mean: {linear_mean:.2f}")
    print(f"  Y_aux mean: {mean_val:.4f}, std: {std_val:.4f}")
    print(f"  Distribution: <0.1: {near_0}, 0.1-0.9: {middle}, >0.9: {near_1}")
    print(f"  Balance score: {balance_score:.3f}")
    
    # Track best balance
    if balance_score > best_balance:
        best_balance = balance_score
        best_weights = weights

plt.tight_layout()
plt.savefig('balanced_probability_search.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("BEST BALANCED WEIGHTS FOUND:")
print("="*60)
if best_weights:
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = best_weights
    print(f"Y_aux = sigmoid({T_coef}*T + {Z1_coef}*Z1 + {Z2_coef}*Z2 + {D_coef}*D + {J_coef}*J + {P_coef}*P + {U_coef}*U)")
    print(f"Balance score: {best_balance:.3f} (fraction of values in 0.1-0.9 range)")
    
    # Test the best weights
    linear_combo = T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U
    Y_aux_best = sigmoid(linear_combo)
    
    print(f"\nWith best weights:")
    print(f"  Linear combination range: [{np.min(linear_combo):.2f}, {np.max(linear_combo):.2f}]")
    print(f"  Y_aux mean: {np.mean(Y_aux_best):.4f}")
    print(f"  Y_aux std: {np.std(Y_aux_best):.4f}")
    print(f"  Values < 0.1: {np.sum(Y_aux_best < 0.1)}")
    print(f"  Values 0.1-0.9: {np.sum((Y_aux_best >= 0.1) & (Y_aux_best <= 0.9))}")
    print(f"  Values > 0.9: {np.sum(Y_aux_best > 0.9)}")