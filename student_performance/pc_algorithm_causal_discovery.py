import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PCAlgorithm:
    """PC Algorithm for causal discovery"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Significance level for independence tests
        self.skeleton = None
        self.dag = None
        self.sepsets = {}
        
    def partial_correlation_test(self, data, x, y, z_set):
        """Test conditional independence X ⊥ Y | Z using partial correlation"""
        if len(z_set) == 0:
            # Marginal correlation
            corr, p_val = pearsonr(data[x], data[y])
            return p_val > self.alpha, p_val
        
        # Partial correlation using regression residuals
        from sklearn.linear_model import LinearRegression
        
        if len(z_set) >= len(data) - 2:
            return True, 1.0  # Not enough data
            
        lr = LinearRegression()
        
        # Regress X on Z
        z_data = data[list(z_set)].values
        lr.fit(z_data, data[x])
        x_residual = data[x] - lr.predict(z_data)
        
        # Regress Y on Z  
        lr.fit(z_data, data[y])
        y_residual = data[y] - lr.predict(z_data)
        
        # Test independence of residuals
        if np.std(x_residual) == 0 or np.std(y_residual) == 0:
            return True, 1.0
            
        corr, p_val = pearsonr(x_residual, y_residual)
        return p_val > self.alpha, p_val
    
    def learn_skeleton(self, data):
        """Learn the skeleton (undirected graph) using PC algorithm"""
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize complete graph
        skeleton = nx.Graph()
        skeleton.add_nodes_from(variables)
        
        # Add all possible edges
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                skeleton.add_edge(variables[i], variables[j])
        
        # PC Algorithm: Remove edges based on conditional independence
        max_cond_set_size = min(n_vars - 2, 4)  # Limit for computational efficiency
        
        for cond_size in range(max_cond_set_size + 1):
            edges_to_remove = []
            
            for edge in list(skeleton.edges()):
                x, y = edge
                
                # Get neighbors of x (excluding y)
                neighbors_x = set(skeleton.neighbors(x)) - {y}
                
                if len(neighbors_x) >= cond_size:
                    # Test all conditioning sets of size cond_size
                    for z_set in combinations(neighbors_x, cond_size):
                        z_set = set(z_set)
                        
                        # Test X ⊥ Y | Z
                        is_independent, p_val = self.partial_correlation_test(data, x, y, z_set)
                        
                        if is_independent:
                            edges_to_remove.append((x, y))
                            self.sepsets[(x, y)] = z_set
                            self.sepsets[(y, x)] = z_set
                            break
            
            # Remove edges
            for edge in edges_to_remove:
                if skeleton.has_edge(edge[0], edge[1]):
                    skeleton.remove_edge(edge[0], edge[1])
        
        self.skeleton = skeleton
        return skeleton
    
    def orient_edges(self, data):
        """Orient edges to form DAG using PC rules"""
        if self.skeleton is None:
            raise ValueError("Must learn skeleton first")
        
        # Convert skeleton to directed graph
        dag = nx.DiGraph()
        dag.add_nodes_from(self.skeleton.nodes())
        
        # Add undirected edges as bidirectional
        for edge in self.skeleton.edges():
            dag.add_edge(edge[0], edge[1])
            dag.add_edge(edge[1], edge[0])
        
        # Rule 1: Orient v-structures (colliders)
        # If X-Z-Y and X,Y not adjacent, and Z not in sepset(X,Y), then X→Z←Y
        for z in dag.nodes():
            neighbors = list(self.skeleton.neighbors(z))
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    x, y = neighbors[i], neighbors[j]
                    
                    # Check if X and Y are not adjacent
                    if not self.skeleton.has_edge(x, y):
                        # Check if Z is not in separation set of X and Y
                        sepset = self.sepsets.get((x, y), set())
                        if z not in sepset:
                            # Orient as X→Z←Y (remove reverse edges)
                            if dag.has_edge(z, x):
                                dag.remove_edge(z, x)
                            if dag.has_edge(z, y):
                                dag.remove_edge(z, y)
        
        # Rule 2: Avoid new v-structures
        # If X→Z-Y and X,Y not adjacent, then Z→Y
        changed = True
        while changed:
            changed = False
            for z in list(dag.nodes()):
                for x in list(dag.predecessors(z)):
                    for y in list(dag.neighbors(z)):
                        if y != x and dag.has_edge(z, y) and dag.has_edge(y, z):
                            if not self.skeleton.has_edge(x, y):
                                dag.remove_edge(y, z)
                                changed = True
        
        # Rule 3: Orient remaining edges to avoid cycles
        changed = True
        while changed:
            changed = False
            for edge in list(dag.edges()):
                x, y = edge
                if dag.has_edge(y, x):  # Bidirectional edge
                    # Try orienting X→Y
                    dag.remove_edge(y, x)
                    if nx.is_directed_acyclic_graph(dag):
                        changed = True
                    else:
                        # Restore and try Y→X
                        dag.add_edge(y, x)
                        dag.remove_edge(x, y)
                        if nx.is_directed_acyclic_graph(dag):
                            changed = True
                        else:
                            # Restore bidirectional
                            dag.add_edge(x, y)
        
        self.dag = dag
        return dag
    
    def fit(self, data):
        """Run complete PC algorithm"""
        print(f"Running PC Algorithm with α={self.alpha}")
        print(f"Data shape: {data.shape}")
        
        # Learn skeleton
        print("Learning skeleton...")
        skeleton = self.learn_skeleton(data)
        print(f"Skeleton: {skeleton.number_of_nodes()} nodes, {skeleton.number_of_edges()} edges")
        
        # Orient edges
        print("Orienting edges...")
        dag = self.orient_edges(data)
        print(f"DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
        
        return dag

def identify_confounders_pc(dag, treatment, outcome):
    """Identify confounders using PC algorithm DAG"""
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
        
        # Confounder: common cause of treatment and outcome
        if causes_treatment and causes_outcome:
            confounders.append(node)
        
        # Mediator: caused by treatment, causes outcome
        elif caused_by_treatment and causes_outcome:
            mediators.append(node)
            
        # Collider: caused by both treatment and outcome
        elif caused_by_treatment and caused_by_outcome:
            colliders.append(node)
    
    return confounders, mediators, colliders

def visualize_pc_dag(dag, treatment, outcome, confounders, mediators, colliders, save_path=None):
    """Visualize PC algorithm DAG"""
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
    
    # Draw DAG
    nx.draw(dag, pos,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=12,
            font_weight='bold',
            arrows=True,
            arrowsize=25,
            edge_color='black',
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
    
    plt.title('PC Algorithm: Causal DAG\nStudent Performance Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PC Algorithm DAG saved: {save_path}")
    
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

def save_pc_results(data, confounders, mediators, colliders, treatment='sex', outcome='G3'):
    """Save PC algorithm results"""
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
    train_data[features].to_csv('data/train_x_1.0.0_pc_algorithm.csv', index=False)
    train_data[outcome].to_csv('data/train_y_1.0.0_pc_algorithm.csv', index=False)
    train_data[treatment].to_csv('data/train_t_1.0.0_pc_algorithm.csv', index=False)
    
    test_data[features].to_csv('data/test_x_1.0.0_pc_algorithm.csv', index=False)
    test_data[outcome].to_csv('data/test_y_1.0.0_pc_algorithm.csv', index=False)
    test_data[treatment].to_csv('data/test_t_1.0.0_pc_algorithm.csv', index=False)
    
    # Save summary
    with open('data/pc_algorithm_summary.txt', 'w') as f:
        f.write("PC ALGORITHM CAUSAL DISCOVERY - STUDENT PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write(f"Outcome: {outcome}\n\n")
        f.write(f"Confounders ({len(confounders)}): {confounders}\n")
        f.write(f"Mediators ({len(mediators)}): {mediators}\n") 
        f.write(f"Colliders ({len(colliders)}): {colliders}\n\n")
        f.write(f"Features for causal inference: {features}\n")
        f.write(f"Train samples: {len(train_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
    
    print(f"PC Algorithm results saved:")
    print(f"  Confounders: {confounders}")
    print(f"  Mediators: {mediators}")
    print(f"  Colliders: {colliders}")
    print(f"  Features: {features}")

def analyze_dag_structure(dag, treatment, outcome):
    """Analyze DAG structure in detail"""
    print(f"\n=== DAG STRUCTURE ANALYSIS ===")
    print(f"Total edges: {dag.number_of_edges()}")
    
    # Check direct relationships
    if dag.has_edge(treatment, outcome):
        print(f"Direct edge: {treatment} → {outcome}")
    if dag.has_edge(outcome, treatment):
        print(f"Direct edge: {outcome} → {treatment}")
    
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

def main():
    """Run PC Algorithm causal discovery"""
    print("=== PC ALGORITHM CAUSAL DISCOVERY ===\n")
    
    # Load data
    data = load_student_data()
    print(f"Loaded data: {data.shape}")
    
    # Run PC Algorithm with more lenient threshold
    pc = PCAlgorithm(alpha=0.1)
    dag = pc.fit(data)
    
    # Identify confounders
    treatment, outcome = 'sex', 'G3'
    confounders, mediators, colliders = identify_confounders_pc(dag, treatment, outcome)
    
    # Analyze DAG structure
    analyze_dag_structure(dag, treatment, outcome)
    
    print(f"\n=== PC ALGORITHM RESULTS ===")
    print(f"Confounders: {confounders}")
    print(f"Mediators: {mediators}")
    print(f"Colliders: {colliders}")
    
    # Visualize
    visualize_pc_dag(dag, treatment, outcome, confounders, mediators, colliders,
                     save_path='student_performance_pc_algorithm_dag.png')
    
    # Also try stricter threshold
    print("\n=== TRYING STRICTER THRESHOLD (α=0.01) ===")
    pc_strict = PCAlgorithm(alpha=0.01)
    dag_strict = pc_strict.fit(data)
    confounders_strict, mediators_strict, colliders_strict = identify_confounders_pc(dag_strict, treatment, outcome)
    print(f"Strict - Confounders: {confounders_strict}, Mediators: {mediators_strict}, Colliders: {colliders_strict}")
    analyze_dag_structure(dag_strict, treatment, outcome)
    
    # Save results
    save_pc_results(data, confounders, mediators, colliders)
    
    return dag, confounders, mediators, colliders

if __name__ == "__main__":
    dag, confounders, mediators, colliders = main()