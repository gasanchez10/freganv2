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

# Test Set 3 weights: [0.2, 0.02, 0.02, 0.008, 0.008, 0.008, 0.003]
T_coef, Z1_coef, Z2_coef, D_coef, J_coef, P_coef, U_coef = [0.2, 0.02, 0.02, 0.008, 0.008, 0.008, 0.003]

# Calculate with Set 3 weights
linear_combo = T_coef*T + Z1_coef*Z1 + Z2_coef*Z2 + D_coef*D + J_coef*J + P_coef*P + U_coef*U
Y_aux_set3 = sigmoid(linear_combo)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Original extreme weights
Y_aux_original = sigmoid(60.0*T + 3.5*Z1 + 3.0*Z2 + 1.5*D + 1.0*J + 0.6*P + 0.1*U)

# Plot original
ax1.hist(Y_aux_original, bins=30, alpha=0.7, color='red', edgecolor='black')
ax1.set_xlabel('Y_aux Probabilities')
ax1.set_ylabel('Frequency')
ax1.set_title('Original Weights (Extreme)')
ax1.grid(True, alpha=0.3)

near_0_orig = np.sum(Y_aux_original < 0.1)
near_1_orig = np.sum(Y_aux_original > 0.9)
middle_orig = np.sum((Y_aux_original >= 0.1) & (Y_aux_original <= 0.9))

ax1.text(0.02, 0.98, f'Mean: {np.mean(Y_aux_original):.3f}\n<0.1: {near_0_orig}\n>0.9: {near_1_orig}\nMid: {middle_orig}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Plot Set 3
ax2.hist(Y_aux_set3, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax2.set_xlabel('Y_aux Probabilities')
ax2.set_ylabel('Frequency')
ax2.set_title('Set 3 Weights (Balanced)')
ax2.grid(True, alpha=0.3)

near_0_set3 = np.sum(Y_aux_set3 < 0.1)
near_1_set3 = np.sum(Y_aux_set3 > 0.9)
middle_set3 = np.sum((Y_aux_set3 >= 0.1) & (Y_aux_set3 <= 0.9))

ax2.text(0.02, 0.98, f'Mean: {np.mean(Y_aux_set3):.3f}\n<0.1: {near_0_set3}\n>0.9: {near_1_set3}\nMid: {middle_set3}', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('set3_weights_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*60)
print("SET 3 WEIGHTS VERIFICATION")
print("="*60)
print(f"Weights: T={T_coef}, Z1={Z1_coef}, Z2={Z2_coef}, D={D_coef}, J={J_coef}, P={P_coef}, U={U_coef}")
print(f"Linear combination range: [{np.min(linear_combo):.2f}, {np.max(linear_combo):.2f}]")
print(f"Y_aux statistics:")
print(f"  Mean: {np.mean(Y_aux_set3):.4f}")
print(f"  Std: {np.std(Y_aux_set3):.4f}")
print(f"  Min: {np.min(Y_aux_set3):.4f}")
print(f"  Max: {np.max(Y_aux_set3):.4f}")
print(f"Distribution:")
print(f"  Values < 0.1: {near_0_set3} ({near_0_set3/N*100:.1f}%)")
print(f"  Values 0.1-0.9: {middle_set3} ({middle_set3/N*100:.1f}%)")
print(f"  Values > 0.9: {near_1_set3} ({near_1_set3/N*100:.1f}%)")
print(f"Balance score: {middle_set3/N:.3f}")

print("\nRECOMMENDED UPDATE:")
print("Replace the line:")
print("Y_aux = sigmoid(60.0*T + 3.5*Z1 + 3.0*Z2 + 1.5*D + 1.0*J + 0.6*P + 0.1*U)")
print("with:")
print("Y_aux = sigmoid(0.2*T + 0.02*Z1 + 0.02*Z2 + 0.008*D + 0.008*J + 0.008*P + 0.003*U)")