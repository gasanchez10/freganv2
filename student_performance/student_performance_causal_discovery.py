import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_student_performance_data():
    """Load student performance dataset from existing processed data"""
    # Load the existing processed data
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
    
    # Define treatment and outcome
    treatment = 'sex'  # Gender as treatment
    outcome = 'G3'     # Final grade as outcome
    
    return data, treatment, outcome

def create_correlation_based_graph(data, treatment, outcome, threshold=0.2):
    """Create causal graph based on correlations"""
    G = nx.DiGraph()
    
    # Add all nodes
    for col in data.columns:
        G.add_node(col)
    
    # Calculate correlations
    corr_matrix = data.corr()
    
    # Add edges based on correlations with treatment and outcome
    for col in data.columns:
        if col != treatment and col != outcome:
            # Edge to treatment if correlated
            if abs(corr_matrix.loc[col, treatment]) > threshold:
                G.add_edge(col, treatment)
            
            # Edge to outcome if correlated
            if abs(corr_matrix.loc[col, outcome]) > threshold:
                G.add_edge(col, outcome)
    
    # Add treatment -> outcome edge
    if abs(corr_matrix.loc[treatment, outcome]) > 0.1:
        G.add_edge(treatment, outcome)
    
    return G

def identify_confounders_mediators(G, treatment, outcome):
    """Identify confounders and mediators from the causal graph"""
    confounders = []
    mediators = []
    
    # Get all nodes except treatment and outcome
    other_nodes = [node for node in G.nodes() if node not in [treatment, outcome]]
    
    for node in other_nodes:
        # Check if node is a confounder
        # Confounder: has edges to both treatment and outcome
        has_edge_to_treatment = G.has_edge(node, treatment)
        has_edge_to_outcome = G.has_edge(node, outcome)
        
        # Check if node is on path from treatment to outcome (potential mediator)
        is_on_treatment_path = G.has_edge(treatment, node) and G.has_edge(node, outcome)
        
        if has_edge_to_treatment and has_edge_to_outcome and not is_on_treatment_path:
            confounders.append(node)
        elif is_on_treatment_path:
            mediators.append(node)
    
    return confounders, mediators

def visualize_causal_graph(G, treatment, outcome, confounders, mediators, save_path=None):
    """Visualize the causal graph with colored nodes"""
    plt.figure(figsize=(15, 12))
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Define colors
    node_colors = []
    for node in G.nodes():
        if node == treatment:
            node_colors.append('green')      # Treatment in green
        elif node == outcome:
            node_colors.append('orange')     # Outcome in orange
        elif node in confounders:
            node_colors.append('red')        # Confounders in red
        elif node in mediators:
            node_colors.append('blue')       # Mediators in blue
        else:
            node_colors.append('lightgray')  # Other variables in light gray
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=1200,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            alpha=0.8,
            with_labels=True)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Treatment (sex)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Outcome (G3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=12, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title('Causal Discovery: Student Performance Dataset\n(Treatment: Gender, Outcome: Final Grade G3)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Causal graph saved to: {save_path}")
    
    plt.show()

def create_train_test_splits(data, test_size=0.2, random_state=42):
    """Create train/test splits like the confounders_only script"""
    # Split the data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['sex'])
    
    return train_data, test_data

def save_causal_discovery_data(train_data, test_data, confounders, mediators, treatment='sex', outcome='G3'):
    """Save data in the same format as confounders_only script"""
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    
    # Prepare confounders-only dataset
    confounder_features = confounders + [treatment]  # Include treatment
    
    # Extract features and labels
    train_x_conf = train_data[confounder_features]
    train_y_conf = train_data[outcome]
    train_t_conf = train_data[treatment]
    
    test_x_conf = test_data[confounder_features]
    test_y_conf = test_data[outcome]
    test_t_conf = test_data[treatment]
    
    # Save confounders-only data
    train_x_conf.to_csv(f'data/train_x_{version}_causal_discovery.csv', index=False)
    train_y_conf.to_csv(f'data/train_y_{version}_causal_discovery.csv', index=False)
    train_t_conf.to_csv(f'data/train_t_{version}_causal_discovery.csv', index=False)
    
    test_x_conf.to_csv(f'data/test_x_{version}_causal_discovery.csv', index=False)
    test_y_conf.to_csv(f'data/test_y_{version}_causal_discovery.csv', index=False)
    test_t_conf.to_csv(f'data/test_t_{version}_causal_discovery.csv', index=False)
    
    print(f"Causal discovery data saved:")
    print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
    print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    print(f"  - Confounders used: {confounders}")
    print(f"  - Mediators identified: {mediators}")
    
    # Save summary
    with open('data/causal_discovery_summary.txt', 'w') as f:
        f.write("CAUSAL DISCOVERY RESULTS - STUDENT PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
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
        f.write(f"  - Features in confounders-only dataset: {len(confounder_features)}\n")
    
    return confounder_features

def main():
    """Main function to run causal discovery analysis"""
    print("=== CAUSAL DISCOVERY ANALYSIS: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Create causal graph using correlation-based approach
    print("\nCreating causal graph using correlation analysis...")
    G = create_correlation_based_graph(data, treatment, outcome)
    print(f"Causal graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Identify confounders and mediators
    confounders, mediators = identify_confounders_mediators(G, treatment, outcome)
    
    print(f"\nCausal Discovery Results:")
    print(f"  - Confounders ({len(confounders)}): {confounders}")
    print(f"  - Mediators ({len(mediators)}): {mediators}")
    
    # Visualize the causal graph
    print("\nGenerating causal graph visualization...")
    visualize_causal_graph(G, treatment, outcome, confounders, mediators, 
                          save_path='student_performance_causal_discovery_graph.png')
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = create_train_test_splits(data)
    
    # Save data for analysis
    print("\nSaving causal discovery datasets...")
    confounder_features = save_causal_discovery_data(train_data, test_data, confounders, mediators)
    
    print(f"\n=== CAUSAL DISCOVERY COMPLETED ===")
    print(f"Graph saved: student_performance_causal_discovery_graph.png")
    print(f"Data saved in: ./data/ folder")
    print(f"Summary saved: ./data/causal_discovery_summary.txt")
    print(f"\nReady for regression analysis with {len(confounders)} identified confounders!")
    
    return confounders, mediators, G

if __name__ == "__main__":
    confounders, mediators, graph = main()