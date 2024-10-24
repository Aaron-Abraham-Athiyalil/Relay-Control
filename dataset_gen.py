import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with constrained ranges
current_values = np.random.uniform(0, 10, 500)
voltage_values = np.random.uniform(220, 230, 500)

# Double the data points to ensure balanced classes
currents = np.concatenate((current_values, current_values))
voltages = np.concatenate((voltage_values, voltage_values))

data = {
    'IR': currents,
    'IY': currents,
    'IB': currents,
    'VR': voltages,
    'VY': voltages,
    'VB': voltages
}

# Diverse logic for relays
data['L1'] = ((currents > 5) & (voltages > 225)).astype(int)
data['L2'] = ((currents <= 5) & (voltages <= 225)).astype(int)
data['L3'] = ((currents > 5) | (voltages > 225)).astype(int)
data['L4'] = ((currents <= 5) & (voltages > 225)).astype(int)
data['L5'] = ((currents > 5) & (voltages <= 225)).astype(int)
data['L6'] = ((currents <= 5) | (voltages <= 225)).astype(int)
data['L7'] = ((currents > 5) & (voltages > 225)).astype(int)
data['L8'] = ((currents <= 5) & (voltages <= 225)).astype(int)

df = pd.DataFrame(data)
df.to_csv('high_quality_synthetic_data.csv', index=False)

print("High-quality dataset created and saved as 'high_quality_synthetic_data.csv'")
