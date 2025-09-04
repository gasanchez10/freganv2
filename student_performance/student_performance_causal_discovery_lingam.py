import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import jarque_bera, kurtosis, skew
from sklearn.decomposition import FastICA
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

class LiNGAM:
    """LiNGAM (Linear Non-Gaussian Acyclic Model) implementation"""
    
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.causal_order_ = None
        self.adjacency_matrix_ = None
        
    def _test_non_gaussianity(self, x):
        """Test if variable is non-Gaussian using multiple tests"""
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(x)
        except:
            jb_p = 1.0
            
        # Kurtosis test (excess kurtosis != 0 for non-Gaussian)
        kurt = abs(kurtosis(x))
        
        # Skewness test (skewness != 0 for non-Gaussian)
        skewness = abs(skew(x))
        
        # Combined non-Gaussianity score
        non_gaussian_score = (1 - jb_p) + kurt/10 + skewness/5
        
        return non_gaussian_score, jb_p < 0.05
    
    def _find_causal_order_ica(self, X):
        """Find causal order using ICA-based approach"""
        n_vars = X.shape[1]
        
        # Apply ICA to find independent components
        ica = FastICA(n_components=n_vars, random_state=42, max_iter=1000)
        try:
            S = ica.fit_transform(X)  # Independent components
            W = ica.components_  # Unmixing matrix
            A = np.linalg.pinv(W)  # Mixing matrix
        except:
            # If ICA fails, use identity
            S = X.copy()
            A = np.eye(n_vars)
        
        # Analyze non-Gaussianity of components
        non_gaussian_scores = []
        for i in range(n_vars):
            score, _ = self._test_non_gaussianity(S[:, i])
            non_gaussian_scores.append(score)
        
        # Order variables by their connection to non-Gaussian components
        # Variables more connected to non-Gaussian components are likely causes
        causal_strength = []
        for i in range(n_vars):
            strength = sum(abs(A[i, j]) * non_gaussian_scores[j] for j in range(n_vars))
            causal_strength.append((strength, i))
        
        # Sort by causal strength (higher = more likely to be a cause)
        causal_strength.sort(reverse=True)
        causal_order = [idx for _, idx in causal_strength]
        
        return causal_order, A, S
    
    def _estimate_adjacency_matrix(self, X, causal_order):
        """Estimate adjacency matrix given causal order"""
        n_vars = X.shape[1]
        B = np.zeros((n_vars, n_vars))
        
        # For each variable in causal order, regress on previous variables
        for i, var_idx in enumerate(causal_order):
            if i == 0:
                continue  # First variable has no parents
                
            # Get potential parents (variables earlier in causal order)
            parent_indices = causal_order[:i]
            
            if len(parent_indices) > 0:
                # Linear regression: X[var_idx] = sum(B[parent, var_idx] * X[parent]) + noise
                X_parents = X[:, parent_indices]
                y = X[:, var_idx]
                
                # Solve least squares
                try:
                    coeffs = np.linalg.lstsq(X_parents, y, rcond=None)[0]
                    
                    # Set coefficients in adjacency matrix
                    for j, parent_idx in enumerate(parent_indices):
                        if abs(coeffs[j]) > self.threshold:
                            B[parent_idx, var_idx] = coeffs[j]
                except:
                    pass
        
        return B
    
    def fit(self, X):
        """Fit LiNGAM model to data"""
        print(f"Starting LiNGAM analysis...")
        print(f"Data shape: {X.shape}")
        
        # Test non-Gaussianity of each variable
        print("\nTesting non-Gaussianity of variables:")
        for i in range(X.shape[1]):
            score, is_non_gaussian = self._test_non_gaussianity(X[:, i])
            print(f"  Variable {i}: Non-Gaussian score = {score:.3f}, Non-Gaussian = {is_non_gaussian}")
        
        # Find causal order using ICA
        print("\nFinding causal order using ICA...")
        causal_order, mixing_matrix, independent_components = self._find_causal_order_ica(X)
        
        print(f"Estimated causal order: {causal_order}")
        
        # Estimate adjacency matrix
        print("\nEstimating adjacency matrix...")
        adjacency_matrix = self._estimate_adjacency_matrix(X, causal_order)
        
        self.causal_order_ = causal_order
        self.adjacency_matrix_ = adjacency_matrix
        
        return adjacency_matrix

def lingam_causal_discovery(data, threshold=0.1):
    """Run LiNGAM algorithm for causal discovery"""
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)
    
    # Run LiNGAM
    lingam = LiNGAM(threshold=threshold)
    B_est = lingam.fit(X)
    
    # Create directed graph
    dag = nx.DiGraph()
    variables = list(data.columns)
    dag.add_nodes_from(variables)
    
    # Add edges based on estimated adjacency matrix
    n_vars = len(variables)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and abs(B_est[i, j]) > 0:
                dag.add_edge(variables[i], variables[j], weight=B_est[i, j])
    
    return dag, B_est, lingam.causal_order_

def identify_confounders_mediators_lingam(dag, treatment, outcome, causal_order, variables):
    """Identify confounders and mediators from LiNGAM result"""
    confounders = []
    mediators = []
    
    # Get indices in causal order
    treatment_idx = variables.index(treatment)
    outcome_idx = variables.index(outcome)
    
    other_nodes = [node for node in dag.nodes() if node not in [treatment, outcome]]
    
    for node in other_nodes:
        node_idx = variables.index(node)
        
        # Confounder: has directed edges to both treatment and outcome
        # AND appears before both in causal order
        affects_treatment = dag.has_edge(node, treatment)
        affects_outcome = dag.has_edge(node, outcome)
        
        # Check causal order position
        node_order_pos = causal_order.index(node_idx)
        treatment_order_pos = causal_order.index(treatment_idx)
        outcome_order_pos = causal_order.index(outcome_idx)
        
        is_before_treatment = node_order_pos < treatment_order_pos
        is_before_outcome = node_order_pos < outcome_order_pos
        
        # Mediator: treatment -> node -> outcome in causal order
        is_mediator = (dag.has_edge(treatment, node) and dag.has_edge(node, outcome) and
                      treatment_order_pos < node_order_pos < outcome_order_pos)
        
        if affects_treatment and affects_outcome and is_before_treatment and is_before_outcome and not is_mediator:
            confounders.append(node)
        elif is_mediator:
            mediators.append(node)
    
    return confounders, mediators

def visualize_lingam_graph(dag, treatment, outcome, confounders, mediators, causal_order, variables, save_path=None):
    """Visualize LiNGAM result with causal ordering"""
    plt.figure(figsize=(18, 14))
    
    # Create hierarchical layout based on causal order
    pos = {}
    n_vars = len(variables)
    
    # Arrange nodes in causal order from left to right
    for i, var_idx in enumerate(causal_order):
        var_name = variables[var_idx]
        x_pos = i * 2  # Horizontal position based on causal order
        
        # Vertical position based on node type
        if var_name == treatment:
            y_pos = 2
        elif var_name == outcome:
            y_pos = 2
        elif var_name in confounders:
            y_pos = 4
        elif var_name in mediators:
            y_pos = 0
        else:
            y_pos = 1
            
        pos[var_name] = (x_pos, y_pos)
    
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
    if edges:
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
    
    # Add causal order annotation
    causal_order_str = " → ".join([variables[i] for i in causal_order])
    plt.text(0.02, 0.98, f"Causal Order: {causal_order_str}", 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=18, label='Treatment (sex)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=18, label='Outcome (G3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=12, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.title('LiNGAM Causal Discovery: Student Performance\n(Linear Non-Gaussian Acyclic Model)', 
              fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LiNGAM graph saved to: {save_path}")
    
    plt.show()

def save_lingam_results(train_data, test_data, confounders, mediators, causal_order, variables, treatment='sex', outcome='G3'):
    """Save LiNGAM results"""
    import os
    
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    suffix = "causal_discovery_lingam"
    
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
    
    print(f"LiNGAM data saved:")
    print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
    print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    print(f"  - Confounders: {confounders}")
    print(f"  - Mediators: {mediators}")
    
    # Save detailed summary
    causal_order_names = [variables[i] for i in causal_order]
    
    with open(f'data/causal_discovery_lingam_summary.txt', 'w') as f:
        f.write("LiNGAM CAUSAL DISCOVERY - STUDENT PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: LiNGAM (Linear Non-Gaussian Acyclic Model)\n")
        f.write("Approach: ICA-based causal ordering with non-Gaussian noise\n")
        f.write("Key Assumption: Linear relations with non-Gaussian noise\n")
        f.write(f"Treatment Variable: {treatment} (Gender)\n")
        f.write(f"Outcome Variable: {outcome} (Final Grade)\n\n")
        f.write(f"Estimated Causal Order:\n")
        f.write(f"  {' → '.join(causal_order_names)}\n\n")
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
    """Main function for LiNGAM causal discovery"""
    print("=== LiNGAM CAUSAL DISCOVERY: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Run LiNGAM Algorithm
    print("\n" + "="*60)
    print("RUNNING LiNGAM ALGORITHM")
    print("="*60)
    print("Method: Linear Non-Gaussian Acyclic Model")
    print("Key assumption: Linear relations with non-Gaussian noise")
    
    dag, B_est, causal_order = lingam_causal_discovery(data, threshold=0.1)
    variables = list(data.columns)
    
    # Identify confounders and mediators
    confounders, mediators = identify_confounders_mediators_lingam(dag, treatment, outcome, causal_order, variables)
    
    print(f"\nLiNGAM Results:")
    print(f"  - DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"  - Non-zero coefficients: {np.count_nonzero(B_est)}")
    print(f"  - Causal order: {[variables[i] for i in causal_order]}")
    print(f"  - Confounders ({len(confounders)}): {confounders}")
    print(f"  - Mediators ({len(mediators)}): {mediators}")
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sex'])
    
    # Visualize results
    print("\nGenerating LiNGAM visualization...")
    visualize_lingam_graph(dag, treatment, outcome, confounders, mediators, causal_order, variables,
                          save_path='student_performance_causal_discovery_lingam.png')
    
    # Save results
    print("\nSaving LiNGAM results...")
    save_lingam_results(train_data, test_data, confounders, mediators, causal_order, variables)
    
    print(f"\n=== LiNGAM CAUSAL DISCOVERY COMPLETED ===")
    print(f"Method: LiNGAM - Linear Non-Gaussian Acyclic Model")
    print(f"Results: {len(confounders)} confounders, {len(mediators)} mediators")
    print(f"Graph saved: student_performance_causal_discovery_lingam.png")
    print(f"Data ready for regression analysis!")
    
    return confounders, mediators, dag, B_est, causal_order

if __name__ == "__main__":
    confounders, mediators, dag, B_est, causal_order = main()