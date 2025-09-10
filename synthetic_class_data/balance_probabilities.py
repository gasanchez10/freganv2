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

# Test different weight combinations
weight_combinations = [
    # [T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef]
    [60.0, 3.5, 3.0, 1.5, 1.0, 0.6, 0.1],  # Original (too extreme)
    [2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],   # Much smaller
    [1.5, 0.4, 0.3, 0.25, 0.15, 0.1, 0.05], # Even smaller
    [1.0, 0.3, 0.25, 0.2, 0.1, 0.08, 0.02], # Smallest
    [3.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],    # Medium
    [0.8, 0.2, 0.15, 0.12, 0.08, 0.05, 0.01] # Very small
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, weights in enumerate(weight_combinations):
    T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = weights
    
    # Calculate Y_aux with current weights
    Y_aux = sigmoid(T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U)
    
    # Plot histogram
    axes[i].hist(Y_aux, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[i].set_xlabel('Y_aux Probabilities')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Weights: T={T_coef}, Z1={Z1_coef}, Z2={Z2_coef}\nD={D_coef}, J={J_coef}, P={P_coef}, U={U_coef}')
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(Y_aux)
    std_val = np.std(Y_aux)
    near_0 = np.sum(Y_aux < 0.1)
    near_1 = np.sum(Y_aux > 0.9)
    middle = np.sum((Y_aux >= 0.1) & (Y_aux <= 0.9))
    
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\n<0.1: {near_0}\n>0.9: {near_1}\nMid: {middle}'
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print(f"\nWeights {i+1}: T={T_coef}, Z1={Z1_coef}, Z2={Z2_coef}, D={D_coef}, J={J_coef}, P={P_coef}, U={U_coef}")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Near 0 (<0.1): {near_0}, Near 1 (>0.9): {near_1}, Middle (0.1-0.9): {middle}")
    print(f"  Balance score (middle/total): {middle/N:.3f}")

plt.tight_layout()
plt.savefig('probability_balance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Recommend best weights
print("\n" + "="*60)
print("RECOMMENDATIONS FOR BALANCED PROBABILITIES:")
print("="*60)
print("For more balanced probabilities (avoiding extremes), use:")
print("Y_aux = sigmoid(1.5*T + 0.4*Z1 + 0.3*Z2 + 0.25*D + 0.15*J + 0.1*P + 0.05*U)")
print("or")
print("Y_aux = sigmoid(1.0*T + 0.3*Z1 + 0.25*Z2 + 0.2*D + 0.1*J + 0.08*P + 0.02*U)")
print("\nThese will give you more probabilities in the 0.1-0.9 range")
print("instead of mostly 0s and 1s.")