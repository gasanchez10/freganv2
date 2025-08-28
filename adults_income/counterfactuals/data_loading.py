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
    names = ["train_x", "train_t", "train_y","test_x", "test_t", "test_y"]
    
    # Load all CSV files into global variables
    for name in names:
        relative_path = f"{base_path}{name}_{version}_continuous.csv"
        globals()[name] = pd.read_csv(relative_path)


    # Return all necessary data components
    return train_x, train_t, train_y, test_x, test_t, test_y