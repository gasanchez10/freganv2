import pandas as pd
import numpy as np

def validate_data_type(data, name):
    """Check if data contains probabilities or binaries"""
    values = data.values.flatten()
    unique_vals = np.unique(values)
    
    print(f"\n{name}:")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  Unique values: {len(unique_vals)}")
    print(f"  Sample values: {values[:10]}")
    
    # Check if binary (only 0s and 1s)
    is_binary = np.all(np.isin(values, [0, 1]))
    print(f"  Is binary (0/1 only): {is_binary}")
    
    # Check if probabilities (between 0 and 1)
    is_prob = np.all((values >= 0) & (values <= 1))
    print(f"  Is probability range [0,1]: {is_prob}")
    
    return is_binary, is_prob

# Load datasets
base_path = "./data/"
train_cf_manual = pd.read_csv(f"{base_path}train_cf_manual_1.0.0_binary2025.csv")
train_cf_random = pd.read_csv(f"{base_path}train_cf_random_1.0.0_binary2025.csv")
train_cf_y = pd.read_csv(f"{base_path}train_cf_y_1.0.0_binary2025.csv")

# Load GANITE counterfactuals
ganite_path = "./counterfactuals/alpha_5/"
test_y_hat = pd.read_csv(f"{ganite_path}test_y_hat_1.0.0_binary_ganite_alpha5.csv")
train_t = pd.read_csv(f"{ganite_path}train_t_1.0.0_binary_ganite_alpha5.csv")
train_size = len(train_t)
train_y_hat = test_y_hat.iloc[:train_size]

# Extract GANITE counterfactuals
train_cf_ganite = []
for index, row in train_y_hat.iterrows():
    cf_treatment = 1 - int(train_t.iloc[index, 0])
    train_cf_ganite.append(row[str(cf_treatment)])
ganite_cf = pd.DataFrame(train_cf_ganite, columns=['0'])

# Load CATE counterfactuals
try:
    train_potential = pd.read_csv("./cate_counterfactuals/cate_results/train_potential_y_1.0.0_binary_cate.csv")
    train_t_cate = pd.read_csv("./data/train_t_1.0.0_binary2025.csv")
    
    train_cf_cate = []
    for i in range(len(train_potential)):
        if train_t_cate.iloc[i, 0] == 0:
            train_cf_cate.append(train_potential.iloc[i, 1])
        else:
            train_cf_cate.append(train_potential.iloc[i, 0])
    cate_cf = pd.DataFrame(train_cf_cate, columns=['0'])
except:
    cate_cf = None

print("COUNTERFACTUAL DATA VALIDATION")
print("="*50)

validate_data_type(train_cf_manual, "Manual Counterfactuals")
validate_data_type(train_cf_random, "Random Counterfactuals") 
validate_data_type(train_cf_y, "True Counterfactuals")
validate_data_type(ganite_cf, "GANITE Counterfactuals")
if cate_cf is not None:
    validate_data_type(cate_cf, "CATE Counterfactuals")