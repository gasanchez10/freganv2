import numpy as np
import matplotlib.pyplot as plt

# Configuration parameters
N = 1000  # Sample size
mu = 0    # Mean for normal distributions
sigma = 3 # Standard deviation
np.random.seed(123)  # Reproducibility

# Sigmoid activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def round_list_items(lst):
    return [round(item) for item in lst]

# Generate exogenous variables (confounders)
Z1 = np.random.normal(mu, 1.5*sigma, N)
Z2 = np.random.normal(mu, sigma, N)

# Generate treatment assignment based on confounders
UT = np.random.normal(mu, sigma, N)
TT = sigmoid(Z1 + UT)
T = round_list_items(TT)
T = np.asarray(T).reshape([N,])

# Generate unobserved error terms
UJ = np.random.normal(mu, sigma, N)
UP = np.random.normal(mu, sigma, N)
UD = np.random.normal(mu, sigma, N)
U = np.random.normal(mu, sigma, N)

# Generate mediator variables following structural equations
J = 3*T - 2*Z1 + 3*Z2 + UJ
P = J - 2*T + 2*Z2 + 0.5*Z1 + UP
D = 2*J + P + 2*T + Z1 + Z2 + UD

# Generate binary outcome variable targeting ATE>=0.35
Y_aux = sigmoid(60.0*T + 3.5*Z1 + 3.0*Z2 + 1.5*D + 1.0*J + 0.6*P + 0.1*U)

# Create histogram
plt.figure(figsize=(12, 8))
plt.hist(Y_aux, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Y_aux (Sigmoid Probabilities)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Y_aux = sigmoid(60.0*T + 3.5*Z1 + 3.0*Z2 + 1.5*D + 1.0*J + 0.6*P + 0.1*U)', fontsize=14)
plt.grid(True, alpha=0.3)

# Add statistics
plt.text(0.02, 0.98, f'Mean: {np.mean(Y_aux):.4f}\nStd: {np.std(Y_aux):.4f}\nMin: {np.min(Y_aux):.4f}\nMax: {np.max(Y_aux):.4f}', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('y_aux_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Y_aux Statistics:")
print(f"Mean: {np.mean(Y_aux):.4f}")
print(f"Std: {np.std(Y_aux):.4f}")
print(f"Min: {np.min(Y_aux):.4f}")
print(f"Max: {np.max(Y_aux):.4f}")
print(f"Values near 0: {np.sum(Y_aux < 0.1)}")
print(f"Values near 1: {np.sum(Y_aux > 0.9)}")
print(f"Values in middle (0.1-0.9): {np.sum((Y_aux >= 0.1) & (Y_aux <= 0.9))}")