import numpy as np
import pandas as pd
import os

def data_loading_synthetic(train_rate=0.8):
    """Load synthetic causal inference data for treatment effect estimation.
    
    Returns:
        Tuple containing training and test data with potential outcomes
    """
    # Data file configuration
    base_path = "../data/"
    version = "1.0.0"
    
    # Define all required data files
    names = ["train_x", "train_t", "train_y", "train_cf_y", "train_cf_manual", "train_cf_random", 
             "test_x", "test_t", "test_y", "test_cf_y", "test_cf_manual", "test_cf_random"]
    
    # Load all CSV files into global variables
    for name in names:
        relative_path = f"{base_path}{name}_{version}_binary2025.csv"
        globals()[name] = pd.read_csv(relative_path)

    # Initialize potential outcome arrays
    potential_0 = []  # Potential outcomes under control (T=0)
    potential_1 = []  # Potential outcomes under treatment (T=1)
    potential_0_test = []
    potential_1_test = []

    # Construct potential outcomes for training data
    # For each sample, combine observed and counterfactual outcomes
    for i in range(len(train_y['0'].values)):
        if train_t['0'].values[i] == 0:  # If received control
            potential_0.append(train_y['0'].values[i])        # Observed outcome
            potential_1.append(train_cf_manual['0'].values[i]) # Counterfactual outcome
        else:  # If received treatment
            potential_0.append(train_cf_manual['0'].values[i]) # Counterfactual outcome
            potential_1.append(train_y['0'].values[i])        # Observed outcome

    # Construct potential outcomes for test data
    for i in range(len(test_y['0'].values)):
        if test_t['0'].values[i] == 0:
            potential_0_test.append(test_y['0'].values[i])
            potential_1_test.append(test_cf_manual['0'].values[i])
        else:
            potential_0_test.append(test_cf_manual['0'].values[i])
            potential_1_test.append(test_y['0'].values[i])

    # Return all necessary data components
    return (train_x, train_t, train_y, np.array([potential_0, potential_1]).T, 
            test_x, np.array([potential_0_test, potential_1_test]).T, 
            test_t, np.append(train_t, test_t, axis=0), test_y, train_cf_manual)

