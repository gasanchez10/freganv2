import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_selection import mutual_info_regression
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

def mutual_information_causal_discovery(data, treatment, outcome, mi_threshold=0.1):
    """Use mutual information for causal discovery"""
    G = nx.DiGraph()
    
    # Add all nodes
    for col in data.columns:
        G.add_node(col)
    
    # Calculate mutual information between all variables
    variables = [col for col in data.columns if col not in [treatment, outcome]]
    
    # Mutual information with treatment
    treatment_mi = {}
    for var in variables:
        mi_score = mutual_info_regression(data[[var]], data[treatment], random_state=42)[0]
        treatment_mi[var] = mi_score
        if mi_score > mi_threshold:
            G.add_edge(var, treatment)
    
    # Mutual information with outcome
    outcome_mi = {}
    for var in variables:
        mi_score = mutual_info_regression(data[[var]], data[outcome], random_state=42)[0]
        outcome_mi[var] = mi_score
        if mi_score > mi_threshold:
            G.add_edge(var, outcome)
    
    # Treatment to outcome
    treatment_outcome_mi = mutual_info_regression(data[[treatment]], data[outcome], random_state=42)[0]
    if treatment_outcome_mi > mi_threshold:
        G.add_edge(treatment, outcome)
    
    return G, treatment_mi, outcome_mi

def identify_confounders_mediators_v2(G, treatment, outcome, data, treatment_mi, outcome_mi):
    """Identify confounders and mediators using mutual information approach"""
    confounders = []
    mediators = []
    
    # Get all nodes except treatment and outcome
    other_nodes = [node for node in G.nodes() if node not in [treatment, outcome]]
    
    for node in other_nodes:
        # Check if node is a confounder (affects both treatment and outcome)
        affects_treatment = G.has_edge(node, treatment)
        affects_outcome = G.has_edge(node, outcome)
        
        # Check if node is affected by treatment (potential mediator)
        affected_by_treatment = G.has_edge(treatment, node)
        
        # Use mutual information scores for better classification
        high_mi_treatment = treatment_mi.get(node, 0) > 0.05
        high_mi_outcome = outcome_mi.get(node, 0) > 0.05
        
        if affects_treatment and affects_outcome and high_mi_treatment and high_mi_outcome:
            confounders.append(node)
        elif affected_by_treatment and affects_outcome:
            mediators.append(node)
    
    return confounders, mediators

def partial_correlation_causal_discovery(data, treatment, outcome, threshold=0.15):
    """Use partial correlations for causal discovery"""
    G = nx.DiGraph()
    
    # Add all nodes
    for col in data.columns:
        G.add_node(col)
    
    # Calculate correlations
    corr_matrix = data.corr()
    
    # Variables excluding treatment and outcome
    variables = [col for col in data.columns if col not in [treatment, outcome]]
    
    # Direct effects on treatment and outcome
    for var in variables:
        # Effect on treatment
        corr_with_treatment = abs(corr_matrix.loc[var, treatment])
        if corr_with_treatment > threshold:
            G.add_edge(var, treatment)
        
        # Effect on outcome
        corr_with_outcome = abs(corr_matrix.loc[var, outcome])
        if corr_with_outcome > threshold:
            G.add_edge(var, outcome)
    
    # Treatment to outcome
    treatment_outcome_corr = abs(corr_matrix.loc[treatment, outcome])
    if treatment_outcome_corr > 0.1:
        G.add_edge(treatment, outcome)
    
    return G

def visualize_causal_graph_v2(G, treatment, outcome, confounders, mediators, method_name, save_path=None):
    """Visualize the causal graph with colored nodes"""
    plt.figure(figsize=(16, 12))
    
    # Create layout with better spacing
    pos = nx.spring_layout(G, k=4, iterations=100, seed=42)
    
    # Define colors
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == treatment:
            node_colors.append('green')      # Treatment in green
            node_sizes.append(1500)
        elif node == outcome:
            node_colors.append('orange')     # Outcome in orange
            node_sizes.append(1500)
        elif node in confounders:
            node_colors.append('red')        # Confounders in red
            node_sizes.append(1200)
        elif node in mediators:
            node_colors.append('blue')       # Mediators in blue
            node_sizes.append(1200)
        else:
            node_colors.append('lightgray')  # Other variables in light gray
            node_sizes.append(800)
    
    # Draw the graph
    nx.draw(G, pos, 
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
    
    plt.title(f'Causal Discovery v2: Student Performance Dataset\n({method_name} Method)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Causal graph saved to: {save_path}")
    
    plt.show()

def create_train_test_splits(data, test_size=0.2, random_state=42):
    """Create train/test splits"""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['sex'])
    return train_data, test_data

def save_causal_discovery_data_v2(train_data, test_data, confounders, mediators, method_name, treatment='sex', outcome='G3'):
    """Save data with method-specific naming"""
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    suffix = f"causal_discovery_v2_{method_name.lower().replace(' ', '_')}"
    
    # Prepare confounders-only dataset
    confounder_features = confounders + [treatment]  # Include treatment
    
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
    
    print(f"Causal discovery v2 ({method_name}) data saved:")
    print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
    print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    print(f"  - Confounders used: {confounders}")
    print(f"  - Mediators identified: {mediators}")
    
    # Save summary
    with open(f'data/causal_discovery_v2_{method_name.lower().replace(" ", "_")}_summary.txt', 'w') as f:
        f.write(f"CAUSAL DISCOVERY V2 RESULTS - STUDENT PERFORMANCE ({method_name})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Method: {method_name}\n")
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
    """Main function to run causal discovery analysis with multiple methods"""
    print("=== CAUSAL DISCOVERY V2 ANALYSIS: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Method 1: Mutual Information
    print("\n" + "="*50)
    print("METHOD 1: MUTUAL INFORMATION CAUSAL DISCOVERY")
    print("="*50)
    
    G_mi, treatment_mi, outcome_mi = mutual_information_causal_discovery(data, treatment, outcome)
    confounders_mi, mediators_mi = identify_confounders_mediators_v2(G_mi, treatment, outcome, data, treatment_mi, outcome_mi)
    
    print(f"Mutual Information Results:")
    print(f"  - Graph: {G_mi.number_of_nodes()} nodes, {G_mi.number_of_edges()} edges")
    print(f"  - Confounders ({len(confounders_mi)}): {confounders_mi}")
    print(f"  - Mediators ({len(mediators_mi)}): {mediators_mi}")
    
    # Method 2: Partial Correlation
    print("\n" + "="*50)
    print("METHOD 2: PARTIAL CORRELATION CAUSAL DISCOVERY")
    print("="*50)
    
    G_pc = partial_correlation_causal_discovery(data, treatment, outcome)
    confounders_pc, mediators_pc = identify_confounders_mediators_v2(G_pc, treatment, outcome, data, {}, {})
    
    print(f"Partial Correlation Results:")
    print(f"  - Graph: {G_pc.number_of_nodes()} nodes, {G_pc.number_of_edges()} edges")
    print(f"  - Confounders ({len(confounders_pc)}): {confounders_pc}")
    print(f"  - Mediators ({len(mediators_pc)}): {mediators_pc}")
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = create_train_test_splits(data)
    
    # Visualize and save results for both methods
    print("\nGenerating visualizations and saving data...")
    
    # Method 1: Mutual Information
    visualize_causal_graph_v2(G_mi, treatment, outcome, confounders_mi, mediators_mi, 
                             "Mutual Information", 
                             save_path='student_performance_causal_discovery_v2_mutual_info.png')
    
    save_causal_discovery_data_v2(train_data, test_data, confounders_mi, mediators_mi, 
                                 "Mutual_Information")
    
    # Method 2: Partial Correlation
    visualize_causal_graph_v2(G_pc, treatment, outcome, confounders_pc, mediators_pc, 
                             "Partial Correlation", 
                             save_path='student_performance_causal_discovery_v2_partial_corr.png')
    
    save_causal_discovery_data_v2(train_data, test_data, confounders_pc, mediators_pc, 
                                 "Partial_Correlation")
    
    print(f"\n=== CAUSAL DISCOVERY V2 COMPLETED ===")
    print(f"Method 1 (Mutual Information): {len(confounders_mi)} confounders, {len(mediators_mi)} mediators")
    print(f"Method 2 (Partial Correlation): {len(confounders_pc)} confounders, {len(mediators_pc)} mediators")
    print(f"Graphs and data saved for both methods!")
    
    return {
        'mutual_info': {'confounders': confounders_mi, 'mediators': mediators_mi, 'graph': G_mi},
        'partial_corr': {'confounders': confounders_pc, 'mediators': mediators_pc, 'graph': G_pc}
    }

if __name__ == "__main__":
    results = main()