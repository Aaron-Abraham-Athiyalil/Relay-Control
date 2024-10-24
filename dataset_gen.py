import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate balanced synthetic data
current_values = np.random.uniform(0, 10, 1000)
voltage_values = np.random.uniform(220, 230, 1000)

data = {
    'IR': current_values,
    'IY': current_values,
    'IB': current_values,
    'VR': voltage_values,
    'VY': voltage_values,
    'VB': voltage_values
}

# More diverse logic for relays
data['L1'] = ((current_values > 5) & (voltage_values > 225)).astype(int)
data['L2'] = ((current_values <= 5) & (voltage_values <= 225)).astype(int)
data['L3'] = ((current_values > 5) | (voltage_values > 225)).astype(int)
data['L4'] = ((current_values <= 5) & (voltage_values > 225)).astype(int)
data['L5'] = ((current_values > 5) & (voltage_values <= 225)).astype(int)
data['L6'] = ((current_values <= 5) | (voltage_values <= 225)).astype(int)
data['L7'] = ((current_values > 5) & (voltage_values > 225)).astype(int)
data['L8'] = ((current_values <= 5) & (voltage_values <= 225)).astype(int)

df = pd.DataFrame(data)
df.to_csv('diverse_synthetic_data.csv', index=False)

print("Diverse dataset created and saved as 'diverse_synthetic_data.csv'")
