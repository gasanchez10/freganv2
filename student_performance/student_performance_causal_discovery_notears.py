import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

def load_student_performance_data():
    """Load student performance dataset from existing processed data"""
    train_x = pd.read_csv('data/train_x_1.0.0_continuous.csv')
    train_y = pd.read_csv('data/train_y_1.0.0_continuous.csv')
    train_t = pd.read_csv('data/train_t_1.0.0_continuous.csv')
    
    test_x = pd.read_csv('data/test_x_1.0.0_continuous.csv')
    test_y = pd.read_csv('data/test_y_1.0.0_continuous.csv')
    test_t = pd.read_csv('data/test_t_1.0.0_continuous.csv')
    
    # Combine train and test data for causal discovery
    all_x = pd.concat([train_x, test_x], ignore_index=True)
    all_y = pd.concat([train_y, test_y], ignore_index=True)
    all_t = pd.concat([train_t, test_t], ignore_index=True)
    
    # Create combined dataset
    data = all_x.copy()
    data['G3'] = all_y.iloc[:, 0]  # Outcome
    data['sex'] = all_t.iloc[:, 0]  # Treatment
    
    treatment = 'sex'
    outcome = 'G3'
    
    return data, treatment, outcome

class NOTEARS:
    """NOTEARS algorithm implementation for causal discovery"""
    
    def __init__(self, lambda1=0.1, lambda2=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16):
        self.lambda1 = lambda1  # L1 penalty
        self.lambda2 = lambda2  # L2 penalty
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        
    def _loss(self, W, X):
        """Compute squared loss"""
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        return loss
    
    def _h(self, W):
        """Compute acyclicity constraint h(W) = tr(e^(W*W)) - d"""
        E = expm(W * W)  # Matrix exponential
        h = np.trace(E) - W.shape[0]
        return h
    
    def _adj(self, w):
        """Convert weight vector to adjacency matrix"""
        d = int(np.sqrt(len(w)))
        return w.reshape([d, d])
    
    def _func(self, w, X, rho, alpha, beta):
        """Objective function for optimization"""
        W = self._adj(w)
        loss = self._loss(W, X)
        h = self._h(W)
        l1_penalty = self.lambda1 * np.abs(W).sum()
        l2_penalty = self.lambda2 * (W ** 2).sum()
        obj = loss + l1_penalty + l2_penalty + alpha * h + 0.5 * rho * h * h
        return obj
    
    def fit(self, X):
        """Fit NOTEARS model to data"""
        n, d = X.shape
        w_est = np.zeros(d * d)  # Flattened adjacency matrix
        
        rho, alpha, h = 1.0, 0.0, np.inf
        
        print(f"Starting NOTEARS optimization...")
        print(f"Data shape: {X.shape}")
        
        for i in range(self.max_iter):
            # Minimize objective function
            w_new, h_new = None, None
            
            while rho < self.rho_max:
                try:
                    # Optimize with current penalty parameters
                    sol = minimize(self._func, w_est, 
                                 args=(X, rho, alpha, 0),
                                 method='L-BFGS-B')
                    w_new = sol.x
                    h_new = self._h(self._adj(w_new))
                    
                    if h_new > 0.25 * h:
                        rho *= 10
                    else:
                        break
                        
                except Exception as e:
                    print(f"Optimization failed: {e}")
                    rho *= 10
                    if rho >= self.rho_max:
                        break
            
            if w_new is None:
                print("Optimization failed completely")
                break
                
            w_est, h = w_new, h_new
            alpha += rho * h
            
            print(f"Iteration {i+1}: h = {h:.6f}, rho = {rho:.2e}")
            
            if h <= self.h_tol:
                print(f"Converged at iteration {i+1}")
                break
        
        # Threshold small weights
        W_est = self._adj(w_est)
        W_est[np.abs(W_est) < 0.3] = 0
        
        return W_est

def notears_causal_discovery(data, lambda1=0.1, lambda2=0.1):
    """Run NOTEARS algorithm for causal discovery"""
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)
    
    # Run NOTEARS
    notears = NOTEARS(lambda1=lambda1, lambda2=lambda2, max_iter=50)
    W_est = notears.fit(X)
    
    # Create directed graph
    dag = nx.DiGraph()
    variables = list(data.columns)
    dag.add_nodes_from(variables)
    
    # Add edges based on estimated adjacency matrix
    n_vars = len(variables)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and abs(W_est[i, j]) > 0:
                dag.add_edge(variables[i], variables[j], weight=W_est[i, j])
    
    return dag, W_est

def identify_confounders_mediators_notears(dag, treatment, outcome):
    """Identify confounders and mediators from NOTEARS result"""
    confounders = []
    mediators = []
    
    other_nodes = [node for node in dag.nodes() if node not in [treatment, outcome]]
    
    for node in other_nodes:
        # Confounder: has directed edges to both treatment and outcome
        affects_treatment = dag.has_edge(node, treatment)
        affects_outcome = dag.has_edge(node, outcome)
        
        # Mediator: treatment -> node -> outcome
        is_mediator = dag.has_edge(treatment, node) and dag.has_edge(node, outcome)
        
        if affects_treatment and affects_outcome and not is_mediator:
            confounders.append(node)
        elif is_mediator:
            mediators.append(node)
    
    return confounders, mediators

def visualize_notears_graph(dag, treatment, outcome, confounders, mediators, save_path=None):
    """Visualize NOTEARS result"""
    plt.figure(figsize=(18, 14))
    
    # Create layout
    pos = nx.spring_layout(dag, k=4, iterations=100, seed=42)
    
    # Define colors and sizes
    node_colors = []
    node_sizes = []
    for node in dag.nodes():
        if node == treatment:
            node_colors.append('green')
            node_sizes.append(2500)
        elif node == outcome:
            node_colors.append('orange')
            node_sizes.append(2500)
        elif node in confounders:
            node_colors.append('red')
            node_sizes.append(1800)
        elif node in mediators:
            node_colors.append('blue')
            node_sizes.append(1800)
        else:
            node_colors.append('lightgray')
            node_sizes.append(1200)
    
    # Draw nodes
    nx.draw_networkx_nodes(dag, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Draw edges with weights
    edges = dag.edges()
    weights = [abs(dag[u][v]['weight']) for u, v in edges]
    
    # Normalize weights for edge thickness
    if weights:
        max_weight = max(weights)
        edge_widths = [3 * w / max_weight for w in weights]
    else:
        edge_widths = [1] * len(edges)
    
    nx.draw_networkx_edges(dag, pos,
                          width=edge_widths,
                          alpha=0.6,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=25)
    
    # Draw labels
    nx.draw_networkx_labels(dag, pos, font_size=11, font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=18, label='Treatment (sex)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=18, label='Outcome (G3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=12, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.title('NOTEARS Causal Discovery: Student Performance\n(Continuous Optimization with DAG Constraint)', 
              fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NOTEARS graph saved to: {save_path}")
    
    plt.show()

def save_notears_results(train_data, test_data, confounders, mediators, treatment='sex', outcome='G3'):
    """Save NOTEARS results"""
    import os
    
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    suffix = "causal_discovery_notears"
    
    # Prepare dataset
    if confounders:
        confounder_features = confounders + [treatment]
    else:
        confounder_features = [treatment]
    
    # Extract features and labels
    train_x_conf = train_data[confounder_features]
    train_y_conf = train_data[outcome]
    train_t_conf = train_data[treatment]
    
    test_x_conf = test_data[confounder_features]
    test_y_conf = test_data[outcome]
    test_t_conf = test_data[treatment]
    
    # Save data
    train_x_conf.to_csv(f'data/train_x_{version}_{suffix}.csv', index=False)
    train_y_conf.to_csv(f'data/train_y_{version}_{suffix}.csv', index=False)
    train_t_conf.to_csv(f'data/train_t_{version}_{suffix}.csv', index=False)
    
    test_x_conf.to_csv(f'data/test_x_{version}_{suffix}.csv', index=False)
    test_y_conf.to_csv(f'data/test_y_{version}_{suffix}.csv', index=False)
    test_t_conf.to_csv(f'data/test_t_{version}_{suffix}.csv', index=False)
    
    print(f"NOTEARS data saved:")
    print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
    print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    print(f"  - Confounders: {confounders}")
    print(f"  - Mediators: {mediators}")
    
    # Save summary
    with open(f'data/causal_discovery_notears_summary.txt', 'w') as f:
        f.write("NOTEARS CAUSAL DISCOVERY - STUDENT PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: NOTEARS (No Tears)\n")
        f.write("Approach: Continuous Optimization with DAG Constraint\n")
        f.write("Key Features: Matrix exponential acyclicity constraint\n")
        f.write(f"Treatment Variable: {treatment} (Gender)\n")
        f.write(f"Outcome Variable: {outcome} (Final Grade)\n\n")
        f.write(f"Identified Confounders ({len(confounders)}):\n")
        for conf in confounders:
            f.write(f"  - {conf}\n")
        f.write(f"\nIdentified Mediators ({len(mediators)}):\n")
        for med in mediators:
            f.write(f"  - {med}\n")
        f.write(f"\nDataset Information:\n")
        f.write(f"  - Total samples: {len(train_data) + len(test_data)}\n")
        f.write(f"  - Training samples: {len(train_data)}\n")
        f.write(f"  - Test samples: {len(test_data)}\n")
        f.write(f"  - Features used: {len(confounder_features)}\n")

def main():
    """Main function for NOTEARS causal discovery"""
    print("=== NOTEARS CAUSAL DISCOVERY: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Run NOTEARS Algorithm
    print("\n" + "="*60)
    print("RUNNING NOTEARS ALGORITHM")
    print("="*60)
    print("Method: Continuous optimization with matrix exponential DAG constraint")
    
    dag, W_est = notears_causal_discovery(data, lambda1=0.1, lambda2=0.1)
    
    # Identify confounders and mediators
    confounders, mediators = identify_confounders_mediators_notears(dag, treatment, outcome)
    
    print(f"\nNOTEARS Results:")
    print(f"  - DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"  - Non-zero weights: {np.count_nonzero(W_est)}")
    print(f"  - Confounders ({len(confounders)}): {confounders}")
    print(f"  - Mediators ({len(mediators)}): {mediators}")
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sex'])
    
    # Visualize results
    print("\nGenerating NOTEARS visualization...")
    visualize_notears_graph(dag, treatment, outcome, confounders, mediators,
                           save_path='student_performance_causal_discovery_notears.png')
    
    # Save results
    print("\nSaving NOTEARS results...")
    save_notears_results(train_data, test_data, confounders, mediators)
    
    print(f"\n=== NOTEARS CAUSAL DISCOVERY COMPLETED ===")
    print(f"Method: NOTEARS - Continuous Optimization with DAG Constraint")
    print(f"Results: {len(confounders)} confounders, {len(mediators)} mediators")
    print(f"Graph saved: student_performance_causal_discovery_notears.png")
    print(f"Data ready for regression analysis!")
    
    return confounders, mediators, dag, W_est

if __name__ == "__main__":
    confounders, mediators, dag, W_est = main()