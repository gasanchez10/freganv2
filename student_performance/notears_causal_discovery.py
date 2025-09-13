import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class NOTEARS:
    """NOTEARS algorithm for causal discovery"""
    
    def __init__(self, lambda1=0.1, loss_type='l2'):
        self.lambda1 = lambda1  # L1 regularization
        self.loss_type = loss_type
        self.W = None
        
    def _loss(self, W, X):
        """Compute loss function"""
        n, d = X.shape
        W = W.reshape([d, d])
        
        if self.loss_type == 'l2':
            # Linear model: X = XW + noise
            M = X @ W
            R = X - M
            loss = 0.5 / n * np.sum(R ** 2)
        else:
            raise ValueError('Unknown loss type')
            
        return loss
    
    def _h(self, W):
        """Acyclicity constraint h(W) = tr(e^(W*W)) - d"""
        d = W.shape[0]
        E = np.linalg.matrix_power(np.eye(d) + W * W / d, d - 1)
        return np.trace(E) - d
    
    def _func(self, w, X, rho, alpha):
        """Objective function for optimization"""
        W = w.reshape([X.shape[1], X.shape[1]])
        loss = self._loss(W, X)
        h_val = self._h(W)
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l1_penalty = self.lambda1 * np.sum(np.abs(W))
        return loss + penalty + l1_penalty
    
    def fit(self, X, max_iter=100, h_tol=1e-8, rho_max=1e+16):
        """Fit NOTEARS model"""
        n, d = X.shape
        w_est = np.zeros(d * d)
        rho, alpha, h = 1.0, 0.0, np.inf
        
        print(f"Running NOTEARS on {n}x{d} data")
        
        for i in range(max_iter):
            # Minimize augmented Lagrangian
            sol = minimize(self._func, w_est, args=(X, rho, alpha), method='L-BFGS-B')
            w_new = sol.x
            
            # Update parameters
            W_new = w_new.reshape([d, d])
            h_new = self._h(W_new)
            
            if h_new > 0.25 * h:
                rho *= 10
            else:
                w_est = w_new
                alpha += rho * h_new
                if h_new <= h_tol or rho >= rho_max:
                    break
                h = h_new
                
            if i % 10 == 0:
                print(f"Iter {i}: h={h_new:.6f}, loss={self._loss(W_new, X):.6f}")
        
        # Threshold small weights
        W_est = w_est.reshape([d, d])
        W_est[np.abs(W_est) < 0.3] = 0
        
        self.W = W_est
        print(f"NOTEARS completed: {np.sum(W_est != 0)} edges found")
        return W_est

def create_dag_from_notears(W, variables):
    """Create NetworkX DAG from NOTEARS weight matrix"""
    dag = nx.DiGraph()
    dag.add_nodes_from(variables)
    
    d = len(variables)
    for i in range(d):
        for j in range(d):
            if W[i, j] != 0:
                dag.add_edge(variables[i], variables[j], weight=W[i, j])
    
    return dag

def identify_confounders_notears(dag, treatment, outcome):
    """Identify confounders from NOTEARS DAG"""
    confounders = []
    mediators = []
    colliders = []
    
    for node in dag.nodes():
        if node in [treatment, outcome]:
            continue
            
        # Check causal relationships
        causes_treatment = dag.has_edge(node, treatment)
        causes_outcome = dag.has_edge(node, outcome)
        caused_by_treatment = dag.has_edge(treatment, node)
        caused_by_outcome = dag.has_edge(outcome, node)
        
        # Confounder: causes both treatment and outcome
        if causes_treatment and causes_outcome:
            confounders.append(node)
        
        # Mediator: caused by treatment, causes outcome
        elif caused_by_treatment and causes_outcome:
            mediators.append(node)
            
        # Collider: caused by both treatment and outcome
        elif caused_by_treatment and caused_by_outcome:
            colliders.append(node)
    
    return confounders, mediators, colliders

def visualize_notears_dag(dag, treatment, outcome, confounders, mediators, colliders, save_path=None):
    """Visualize NOTEARS DAG"""
    plt.figure(figsize=(16, 12))
    
    # Create layout
    pos = {}
    pos[treatment] = (0, 0)
    pos[outcome] = (6, 0)
    
    # Position confounders
    if confounders:
        for i, conf in enumerate(confounders):
            pos[conf] = (3, 3 + i * 0.8)
    
    # Position mediators
    if mediators:
        for i, med in enumerate(mediators):
            pos[med] = (3, -1 - i * 0.8)
    
    # Position colliders
    if colliders:
        for i, coll in enumerate(colliders):
            pos[coll] = (3, -3 - i * 0.8)
    
    # Position other nodes
    other_nodes = [n for n in dag.nodes() if n not in [treatment, outcome] + confounders + mediators + colliders]
    if other_nodes:
        for i, node in enumerate(other_nodes):
            angle = 2 * np.pi * i / len(other_nodes)
            pos[node] = (8 + 2 * np.cos(angle), 2 * np.sin(angle))
    
    # Node colors and sizes
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
            node_sizes.append(2000)
        elif node in mediators:
            node_colors.append('blue')
            node_sizes.append(2000)
        elif node in colliders:
            node_colors.append('purple')
            node_sizes.append(1500)
        else:
            node_colors.append('lightgray')
            node_sizes.append(1000)
    
    # Draw DAG with edge weights
    edges = dag.edges()
    weights = [abs(dag[u][v]['weight']) * 3 for u, v in edges]  # Scale edge thickness
    
    nx.draw(dag, pos,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='black',
            width=weights,
            alpha=0.8,
            with_labels=True)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=20, label='Treatment'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=20, label='Outcome'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=18, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=18, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=15, label='Colliders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=12, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.title('NOTEARS: Causal DAG\nStudent Performance Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NOTEARS DAG saved: {save_path}")
    
    plt.show()

def load_student_data():
    """Load student performance data"""
    train_x = pd.read_csv('data/train_x_1.0.0_continuous.csv')
    train_y = pd.read_csv('data/train_y_1.0.0_continuous.csv') 
    train_t = pd.read_csv('data/train_t_1.0.0_continuous.csv')
    
    test_x = pd.read_csv('data/test_x_1.0.0_continuous.csv')
    test_y = pd.read_csv('data/test_y_1.0.0_continuous.csv')
    test_t = pd.read_csv('data/test_t_1.0.0_continuous.csv')
    
    # Combine data
    all_x = pd.concat([train_x, test_x], ignore_index=True)
    all_y = pd.concat([train_y, test_y], ignore_index=True)
    all_t = pd.concat([train_t, test_t], ignore_index=True)
    
    data = all_x.copy()
    data['G3'] = all_y.iloc[:, 0]
    data['sex'] = all_t.iloc[:, 0]
    
    return data

def analyze_dag_structure(dag, treatment, outcome):
    """Analyze DAG structure in detail"""
    print(f"\n=== DAG STRUCTURE ANALYSIS ===")
    print(f"Total edges: {dag.number_of_edges()}")
    
    # Check direct relationships
    if dag.has_edge(treatment, outcome):
        weight = dag[treatment][outcome]['weight']
        print(f"Direct edge: {treatment} → {outcome} (weight: {weight:.3f})")
    if dag.has_edge(outcome, treatment):
        weight = dag[outcome][treatment]['weight']
        print(f"Direct edge: {outcome} → {treatment} (weight: {weight:.3f})")
    
    # Analyze each node's relationships
    for node in dag.nodes():
        if node in [treatment, outcome]:
            continue
        
        predecessors = list(dag.predecessors(node))
        successors = list(dag.successors(node))
        
        if predecessors or successors:
            print(f"{node}: predecessors={predecessors}, successors={successors}")
            
            if treatment in predecessors and outcome in successors:
                print(f"  → MEDIATOR: {treatment} → {node} → {outcome}")
            elif treatment in successors and outcome in successors:
                print(f"  → CONFOUNDER: {node} → {treatment}, {node} → {outcome}")
            elif treatment in predecessors and outcome in predecessors:
                print(f"  → COLLIDER: {treatment} → {node} ← {outcome}")

def save_notears_results(data, confounders, mediators, colliders, treatment='sex', outcome='G3'):
    """Save NOTEARS results"""
    import os
    os.makedirs('data', exist_ok=True)
    
    # Create train/test splits
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[treatment])
    
    # Save confounder-adjusted data
    if confounders:
        features = confounders + [treatment]
    else:
        features = [treatment]
    
    # Save datasets
    train_data[features].to_csv('data/train_x_1.0.0_notears.csv', index=False)
    train_data[outcome].to_csv('data/train_y_1.0.0_notears.csv', index=False)
    train_data[treatment].to_csv('data/train_t_1.0.0_notears.csv', index=False)
    
    test_data[features].to_csv('data/test_x_1.0.0_notears.csv', index=False)
    test_data[outcome].to_csv('data/test_y_1.0.0_notears.csv', index=False)
    test_data[treatment].to_csv('data/test_t_1.0.0_notears.csv', index=False)
    
    # Save summary
    with open('data/notears_summary.txt', 'w') as f:
        f.write("NOTEARS CAUSAL DISCOVERY - STUDENT PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write(f"Outcome: {outcome}\n\n")
        f.write(f"Confounders ({len(confounders)}): {confounders}\n")
        f.write(f"Mediators ({len(mediators)}): {mediators}\n") 
        f.write(f"Colliders ({len(colliders)}): {colliders}\n\n")
        f.write(f"Features for causal inference: {features}\n")
        f.write(f"Train samples: {len(train_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
    
    print(f"NOTEARS results saved:")
    print(f"  Confounders: {confounders}")
    print(f"  Mediators: {mediators}")
    print(f"  Colliders: {colliders}")
    print(f"  Features: {features}")

def main():
    """Run NOTEARS causal discovery"""
    print("=== NOTEARS CAUSAL DISCOVERY ===\n")
    
    # Load data
    data = load_student_data()
    print(f"Loaded data: {data.shape}")
    
    # Standardize data for NOTEARS
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)
    variables = list(data.columns)
    
    # Run NOTEARS
    notears = NOTEARS(lambda1=0.1)
    W = notears.fit(X)
    
    # Create DAG
    dag = create_dag_from_notears(W, variables)
    
    # Identify confounders
    treatment, outcome = 'sex', 'G3'
    confounders, mediators, colliders = identify_confounders_notears(dag, treatment, outcome)
    
    print(f"\n=== NOTEARS RESULTS ===")
    print(f"Confounders: {confounders}")
    print(f"Mediators: {mediators}")
    print(f"Colliders: {colliders}")
    
    # Analyze structure
    analyze_dag_structure(dag, treatment, outcome)
    
    # Visualize
    visualize_notears_dag(dag, treatment, outcome, confounders, mediators, colliders,
                         save_path='student_performance_notears_dag.png')
    
    # Save results
    save_notears_results(data, confounders, mediators, colliders)
    
    return dag, confounders, mediators, colliders

if __name__ == "__main__":
    dag, confounders, mediators, colliders = main()