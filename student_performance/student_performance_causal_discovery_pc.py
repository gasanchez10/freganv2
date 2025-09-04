import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
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

def conditional_independence_test(data, x, y, z_set=None, alpha=0.05):
    """Test conditional independence between x and y given z_set"""
    if z_set is None or len(z_set) == 0:
        # Marginal independence test
        corr, p_value = pearsonr(data[x], data[y])
        return p_value > alpha, p_value
    
    # Partial correlation test
    try:
        # Create design matrix with conditioning variables
        if len(z_set) == 1:
            z = z_set[0]
            # Partial correlation formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
            r_xy, _ = pearsonr(data[x], data[y])
            r_xz, _ = pearsonr(data[x], data[z])
            r_yz, _ = pearsonr(data[y], data[z])
            
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            if abs(denominator) < 1e-10:
                return True, 1.0  # Independence if denominator is zero
            
            partial_corr = (r_xy - r_xz * r_yz) / denominator
            
            # Convert to p-value (approximate)
            n = len(data)
            df = n - len(z_set) - 2
            if df <= 0:
                return True, 1.0
            
            t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2))
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
            
        else:
            # Multiple conditioning variables - use regression residuals
            from sklearn.linear_model import LinearRegression
            
            # Regress x on z_set
            reg_x = LinearRegression()
            reg_x.fit(data[list(z_set)], data[x])
            residuals_x = data[x] - reg_x.predict(data[list(z_set)])
            
            # Regress y on z_set
            reg_y = LinearRegression()
            reg_y.fit(data[list(z_set)], data[y])
            residuals_y = data[y] - reg_y.predict(data[list(z_set)])
            
            # Test independence of residuals
            corr, p_value = pearsonr(residuals_x, residuals_y)
        
        return p_value > alpha, p_value
        
    except Exception as e:
        # If test fails, assume independence
        return True, 1.0

def pc_algorithm(data, alpha=0.05, max_conditioning_size=3):
    """Simplified PC Algorithm implementation"""
    variables = list(data.columns)
    n_vars = len(variables)
    
    # Step 1: Start with complete undirected graph
    skeleton = nx.Graph()
    skeleton.add_nodes_from(variables)
    
    # Add all possible edges
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            skeleton.add_edge(variables[i], variables[j])
    
    print(f"Starting PC algorithm with {skeleton.number_of_nodes()} nodes and {skeleton.number_of_edges()} edges")
    
    # Step 2: Remove edges based on conditional independence
    removed_edges = []
    
    # Start with conditioning sets of size 0, then 1, 2, etc.
    for cond_size in range(min(max_conditioning_size + 1, n_vars - 2)):
        edges_to_remove = []
        
        for edge in list(skeleton.edges()):
            x, y = edge
            
            # Get potential conditioning sets (neighbors of x and y, excluding x and y)
            neighbors_x = set(skeleton.neighbors(x)) - {y}
            neighbors_y = set(skeleton.neighbors(y)) - {x}
            potential_cond_vars = neighbors_x.union(neighbors_y)
            
            # Test all conditioning sets of current size
            if len(potential_cond_vars) >= cond_size:
                for cond_set in combinations(potential_cond_vars, cond_size):
                    is_independent, p_val = conditional_independence_test(data, x, y, cond_set, alpha)
                    
                    if is_independent:
                        edges_to_remove.append((x, y))
                        removed_edges.append((x, y, cond_set, p_val))
                        break  # Found independence, remove edge
        
        # Remove edges found to be conditionally independent
        for edge in edges_to_remove:
            if skeleton.has_edge(edge[0], edge[1]):
                skeleton.remove_edge(edge[0], edge[1])
        
        print(f"Conditioning size {cond_size}: Removed {len(edges_to_remove)} edges, {skeleton.number_of_edges()} remaining")
    
    # Step 3: Orient edges (simplified - just create directed graph)
    dag = nx.DiGraph()
    dag.add_nodes_from(skeleton.nodes())
    
    # Simple orientation: use correlation direction as heuristic
    for edge in skeleton.edges():
        x, y = edge
        corr, _ = pearsonr(data[x], data[y])
        
        # Orient based on correlation strength and variable names
        # Heuristic: variables with lower indices tend to be causes
        var_indices = {var: i for i, var in enumerate(variables)}
        
        if var_indices[x] < var_indices[y]:
            dag.add_edge(x, y)
        else:
            dag.add_edge(y, x)
    
    print(f"Final DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} directed edges")
    
    return dag, skeleton, removed_edges

def identify_confounders_mediators_pc(dag, treatment, outcome):
    """Identify confounders and mediators from PC algorithm result"""
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

def visualize_pc_graph(dag, treatment, outcome, confounders, mediators, save_path=None):
    """Visualize PC algorithm result"""
    plt.figure(figsize=(16, 12))
    
    # Create layout
    pos = nx.spring_layout(dag, k=3, iterations=100, seed=42)
    
    # Define colors
    node_colors = []
    node_sizes = []
    for node in dag.nodes():
        if node == treatment:
            node_colors.append('green')
            node_sizes.append(2000)
        elif node == outcome:
            node_colors.append('orange')
            node_sizes.append(2000)
        elif node in confounders:
            node_colors.append('red')
            node_sizes.append(1500)
        elif node in mediators:
            node_colors.append('blue')
            node_sizes.append(1500)
        else:
            node_colors.append('lightgray')
            node_sizes.append(1000)
    
    # Draw the graph
    nx.draw(dag, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=25,
            edge_color='gray',
            alpha=0.8,
            with_labels=True)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='Treatment (sex)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='Outcome (G3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    plt.title('PC Algorithm Causal Discovery: Student Performance\n(Peter-Clark Algorithm with Conditional Independence)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PC algorithm graph saved to: {save_path}")
    
    plt.show()

def save_pc_results(train_data, test_data, confounders, mediators, treatment='sex', outcome='G3'):
    """Save PC algorithm results"""
    import os
    
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    suffix = "causal_discovery_pc_algorithm"
    
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
    
    print(f"PC Algorithm data saved:")
    print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
    print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    print(f"  - Confounders: {confounders}")
    print(f"  - Mediators: {mediators}")
    
    # Save summary
    with open(f'data/causal_discovery_pc_algorithm_summary.txt', 'w') as f:
        f.write("PC ALGORITHM CAUSAL DISCOVERY - STUDENT PERFORMANCE\n")
        f.write("=" * 55 + "\n\n")
        f.write("Method: Peter-Clark (PC) Algorithm\n")
        f.write("Approach: Conditional Independence Testing + Edge Orientation\n")
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
    """Main function for PC Algorithm causal discovery"""
    print("=== PC ALGORITHM CAUSAL DISCOVERY: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Run PC Algorithm
    print("\n" + "="*60)
    print("RUNNING PC ALGORITHM (PETER-CLARK)")
    print("="*60)
    
    dag, skeleton, removed_edges = pc_algorithm(data, alpha=0.05, max_conditioning_size=2)
    
    # Identify confounders and mediators
    confounders, mediators = identify_confounders_mediators_pc(dag, treatment, outcome)
    
    print(f"\nPC Algorithm Results:")
    print(f"  - Final DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    print(f"  - Edges removed: {len(removed_edges)}")
    print(f"  - Confounders ({len(confounders)}): {confounders}")
    print(f"  - Mediators ({len(mediators)}): {mediators}")
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sex'])
    
    # Visualize results
    print("\nGenerating PC algorithm visualization...")
    visualize_pc_graph(dag, treatment, outcome, confounders, mediators,
                      save_path='student_performance_causal_discovery_pc_algorithm.png')
    
    # Save results
    print("\nSaving PC algorithm results...")
    save_pc_results(train_data, test_data, confounders, mediators)
    
    print(f"\n=== PC ALGORITHM CAUSAL DISCOVERY COMPLETED ===")
    print(f"Method: Peter-Clark Algorithm with Conditional Independence")
    print(f"Results: {len(confounders)} confounders, {len(mediators)} mediators")
    print(f"Graph saved: student_performance_causal_discovery_pc_algorithm.png")
    print(f"Data ready for regression analysis!")
    
    return confounders, mediators, dag, skeleton

if __name__ == "__main__":
    confounders, mediators, dag, skeleton = main()