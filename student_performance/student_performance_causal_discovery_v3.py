import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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

def predictive_causal_discovery(data, treatment, outcome, importance_threshold=0.01):
    """Use predictive importance for causal discovery"""
    G = nx.DiGraph()
    
    # Add all nodes
    for col in data.columns:
        G.add_node(col)
    
    variables = [col for col in data.columns if col not in [treatment, outcome]]
    
    # Feature importance for predicting treatment
    treatment_importances = {}
    if len(variables) > 0:
        rf_treatment = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_treatment.fit(data[variables], data[treatment])
        
        for i, var in enumerate(variables):
            importance = rf_treatment.feature_importances_[i]
            treatment_importances[var] = importance
            if importance > importance_threshold:
                G.add_edge(var, treatment)
    
    # Feature importance for predicting outcome
    outcome_importances = {}
    if len(variables) > 0:
        rf_outcome = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_outcome.fit(data[variables], data[outcome])
        
        for i, var in enumerate(variables):
            importance = rf_outcome.feature_importances_[i]
            outcome_importances[var] = importance
            if importance > importance_threshold:
                G.add_edge(var, outcome)
    
    # Treatment to outcome
    rf_direct = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_direct.fit(data[[treatment]], data[outcome])
    direct_score = rf_direct.score(data[[treatment]], data[outcome])
    if direct_score > 0.01:
        G.add_edge(treatment, outcome)
    
    return G, treatment_importances, outcome_importances

def statistical_causal_discovery(data, treatment, outcome, corr_threshold=0.1, p_threshold=0.05):
    """Use statistical tests for causal discovery"""
    G = nx.DiGraph()
    
    # Add all nodes
    for col in data.columns:
        G.add_node(col)
    
    variables = [col for col in data.columns if col not in [treatment, outcome]]
    
    treatment_stats = {}
    outcome_stats = {}
    
    # Statistical relationships with treatment
    for var in variables:
        # Pearson correlation
        corr, p_val = pearsonr(data[var], data[treatment])
        treatment_stats[var] = {'corr': corr, 'p_val': p_val}
        
        if abs(corr) > corr_threshold and p_val < p_threshold:
            G.add_edge(var, treatment)
    
    # Statistical relationships with outcome
    for var in variables:
        # Pearson correlation
        corr, p_val = pearsonr(data[var], data[outcome])
        outcome_stats[var] = {'corr': corr, 'p_val': p_val}
        
        if abs(corr) > corr_threshold and p_val < p_threshold:
            G.add_edge(var, outcome)
    
    # Treatment to outcome
    treatment_outcome_corr, treatment_outcome_p = pearsonr(data[treatment], data[outcome])
    if abs(treatment_outcome_corr) > 0.05 and treatment_outcome_p < p_threshold:
        G.add_edge(treatment, outcome)
    
    return G, treatment_stats, outcome_stats

def identify_confounders_mediators_v3(G, treatment, outcome, method_stats=None):
    """Enhanced confounder and mediator identification"""
    confounders = []
    mediators = []
    colliders = []
    
    # Get all nodes except treatment and outcome
    other_nodes = [node for node in G.nodes() if node not in [treatment, outcome]]
    
    for node in other_nodes:
        # Check relationships
        affects_treatment = G.has_edge(node, treatment)
        affects_outcome = G.has_edge(node, outcome)
        affected_by_treatment = G.has_edge(treatment, node)
        affected_by_outcome = G.has_edge(outcome, node)
        
        # Confounder: affects both treatment and outcome (backdoor path)
        if affects_treatment and affects_outcome:
            confounders.append(node)
        
        # Mediator: affected by treatment and affects outcome (frontdoor path)
        elif affected_by_treatment and affects_outcome:
            mediators.append(node)
        
        # Collider: affected by both treatment and outcome
        elif affected_by_treatment and affected_by_outcome:
            colliders.append(node)
    
    return confounders, mediators, colliders

def visualize_causal_graph_v3(G, treatment, outcome, confounders, mediators, colliders, method_name, save_path=None):
    """Enhanced visualization with colliders"""
    plt.figure(figsize=(18, 14))
    
    # Create hierarchical layout
    pos = {}
    
    # Position treatment and outcome
    pos[treatment] = (0, 0)
    pos[outcome] = (4, 0)
    
    # Position confounders above
    if confounders:
        for i, conf in enumerate(confounders):
            pos[conf] = (2, 2 + i * 0.5)
    
    # Position mediators in the middle
    if mediators:
        for i, med in enumerate(mediators):
            pos[med] = (2, -0.5 - i * 0.5)
    
    # Position colliders below
    if colliders:
        for i, coll in enumerate(colliders):
            pos[coll] = (2, -2 - i * 0.5)
    
    # Position other nodes
    other_nodes = [node for node in G.nodes() if node not in [treatment, outcome] + confounders + mediators + colliders]
    if other_nodes:
        for i, node in enumerate(other_nodes):
            angle = 2 * np.pi * i / len(other_nodes)
            pos[node] = (6 + 2 * np.cos(angle), 2 * np.sin(angle))
    
    # Define colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
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
        elif node in colliders:
            node_colors.append('purple')
            node_sizes.append(1200)
        else:
            node_colors.append('lightgray')
            node_sizes.append(800)
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            font_size=11,
            font_weight='bold',
            arrows=True,
            arrowsize=30,
            edge_color='gray',
            alpha=0.8,
            with_labels=True)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=18, label='Treatment (sex)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=18, label='Outcome (G3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Confounders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=15, label='Mediators'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=12, label='Colliders'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.title(f'Enhanced Causal Discovery: Student Performance\n({method_name} Method)', 
              fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced causal graph saved to: {save_path}")
    
    plt.show()

def save_enhanced_results(train_data, test_data, confounders, mediators, colliders, method_name, treatment='sex', outcome='G3'):
    """Save enhanced results with all variable types"""
    import os
    
    os.makedirs('data', exist_ok=True)
    
    version = "1.0.0"
    suffix = f"causal_discovery_v3_{method_name.lower().replace(' ', '_')}"
    
    # Create different datasets
    # 1. Confounders only
    if confounders:
        confounder_features = confounders + [treatment]
        
        train_x_conf = train_data[confounder_features]
        train_y_conf = train_data[outcome]
        train_t_conf = train_data[treatment]
        
        test_x_conf = test_data[confounder_features]
        test_y_conf = test_data[outcome]
        test_t_conf = test_data[treatment]
        
        # Save confounders-only data
        train_x_conf.to_csv(f'data/train_x_{version}_{suffix}.csv', index=False)
        train_y_conf.to_csv(f'data/train_y_{version}_{suffix}.csv', index=False)
        train_t_conf.to_csv(f'data/train_t_{version}_{suffix}.csv', index=False)
        
        test_x_conf.to_csv(f'data/test_x_{version}_{suffix}.csv', index=False)
        test_y_conf.to_csv(f'data/test_y_{version}_{suffix}.csv', index=False)
        test_t_conf.to_csv(f'data/test_t_{version}_{suffix}.csv', index=False)
        
        print(f"Enhanced causal discovery ({method_name}) data saved:")
        print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
        print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    else:
        # If no confounders, save treatment only
        train_x_conf = train_data[[treatment]]
        train_y_conf = train_data[outcome]
        train_t_conf = train_data[treatment]
        
        test_x_conf = test_data[[treatment]]
        test_y_conf = test_data[outcome]
        test_t_conf = test_data[treatment]
        
        train_x_conf.to_csv(f'data/train_x_{version}_{suffix}.csv', index=False)
        train_y_conf.to_csv(f'data/train_y_{version}_{suffix}.csv', index=False)
        train_t_conf.to_csv(f'data/train_t_{version}_{suffix}.csv', index=False)
        
        test_x_conf.to_csv(f'data/test_x_{version}_{suffix}.csv', index=False)
        test_y_conf.to_csv(f'data/test_y_{version}_{suffix}.csv', index=False)
        test_t_conf.to_csv(f'data/test_t_{version}_{suffix}.csv', index=False)
        
        print(f"Enhanced causal discovery ({method_name}) data saved (no confounders found):")
        print(f"  - Train: {train_x_conf.shape[0]} samples, {train_x_conf.shape[1]} features")
        print(f"  - Test: {test_x_conf.shape[0]} samples, {test_x_conf.shape[1]} features")
    
    print(f"  - Confounders: {confounders}")
    print(f"  - Mediators: {mediators}")
    print(f"  - Colliders: {colliders}")
    
    # Save detailed summary
    with open(f'data/causal_discovery_v3_{method_name.lower().replace(" ", "_")}_summary.txt', 'w') as f:
        f.write(f"ENHANCED CAUSAL DISCOVERY V3 - STUDENT PERFORMANCE ({method_name})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Treatment Variable: {treatment} (Gender)\n")
        f.write(f"Outcome Variable: {outcome} (Final Grade)\n\n")
        f.write(f"Identified Confounders ({len(confounders)}):\n")
        for conf in confounders:
            f.write(f"  - {conf}\n")
        f.write(f"\nIdentified Mediators ({len(mediators)}):\n")
        for med in mediators:
            f.write(f"  - {med}\n")
        f.write(f"\nIdentified Colliders ({len(colliders)}):\n")
        for coll in colliders:
            f.write(f"  - {coll}\n")
        f.write(f"\nDataset Information:\n")
        f.write(f"  - Total samples: {len(train_data) + len(test_data)}\n")
        f.write(f"  - Training samples: {len(train_data)}\n")
        f.write(f"  - Test samples: {len(test_data)}\n")
        f.write(f"  - Features used: {len(confounders) + 1}\n")

def main():
    """Main function for enhanced causal discovery"""
    print("=== ENHANCED CAUSAL DISCOVERY V3: STUDENT PERFORMANCE ===\n")
    
    # Load data
    print("Loading student performance data...")
    data, treatment, outcome = load_student_performance_data()
    print(f"Data shape: {data.shape}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    
    # Method 1: Predictive Importance
    print("\n" + "="*60)
    print("METHOD 1: PREDICTIVE IMPORTANCE CAUSAL DISCOVERY")
    print("="*60)
    
    G_pred, treatment_imp, outcome_imp = predictive_causal_discovery(data, treatment, outcome)
    confounders_pred, mediators_pred, colliders_pred = identify_confounders_mediators_v3(G_pred, treatment, outcome)
    
    print(f"Predictive Importance Results:")
    print(f"  - Graph: {G_pred.number_of_nodes()} nodes, {G_pred.number_of_edges()} edges")
    print(f"  - Confounders ({len(confounders_pred)}): {confounders_pred}")
    print(f"  - Mediators ({len(mediators_pred)}): {mediators_pred}")
    print(f"  - Colliders ({len(colliders_pred)}): {colliders_pred}")
    
    # Method 2: Statistical Tests
    print("\n" + "="*60)
    print("METHOD 2: STATISTICAL TESTS CAUSAL DISCOVERY")
    print("="*60)
    
    G_stat, treatment_stats, outcome_stats = statistical_causal_discovery(data, treatment, outcome)
    confounders_stat, mediators_stat, colliders_stat = identify_confounders_mediators_v3(G_stat, treatment, outcome)
    
    print(f"Statistical Tests Results:")
    print(f"  - Graph: {G_stat.number_of_nodes()} nodes, {G_stat.number_of_edges()} edges")
    print(f"  - Confounders ({len(confounders_stat)}): {confounders_stat}")
    print(f"  - Mediators ({len(mediators_stat)}): {mediators_stat}")
    print(f"  - Colliders ({len(colliders_stat)}): {colliders_stat}")
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    train_data, test_data = create_train_test_splits(data)
    
    # Visualize and save results
    print("\nGenerating enhanced visualizations and saving data...")
    
    # Method 1: Predictive Importance
    visualize_causal_graph_v3(G_pred, treatment, outcome, confounders_pred, mediators_pred, colliders_pred,
                             "Predictive Importance", 
                             save_path='student_performance_causal_discovery_v3_predictive.png')
    
    save_enhanced_results(train_data, test_data, confounders_pred, mediators_pred, colliders_pred,
                         "Predictive_Importance")
    
    # Method 2: Statistical Tests
    visualize_causal_graph_v3(G_stat, treatment, outcome, confounders_stat, mediators_stat, colliders_stat,
                             "Statistical Tests", 
                             save_path='student_performance_causal_discovery_v3_statistical.png')
    
    save_enhanced_results(train_data, test_data, confounders_stat, mediators_stat, colliders_stat,
                         "Statistical_Tests")
    
    print(f"\n=== ENHANCED CAUSAL DISCOVERY V3 COMPLETED ===")
    print(f"Predictive Method: {len(confounders_pred)} confounders, {len(mediators_pred)} mediators, {len(colliders_pred)} colliders")
    print(f"Statistical Method: {len(confounders_stat)} confounders, {len(mediators_stat)} mediators, {len(colliders_stat)} colliders")
    print(f"Enhanced graphs and datasets saved!")
    
    return {
        'predictive': {'confounders': confounders_pred, 'mediators': mediators_pred, 'colliders': colliders_pred, 'graph': G_pred},
        'statistical': {'confounders': confounders_stat, 'mediators': mediators_stat, 'colliders': colliders_stat, 'graph': G_stat}
    }

def create_train_test_splits(data, test_size=0.2, random_state=42):
    """Create train/test splits"""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['sex'])
    return train_data, test_data

if __name__ == "__main__":
    results = main()