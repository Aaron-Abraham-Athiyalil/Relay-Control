import pandas as pd
import joblib  # Import joblib to load models

# Load the trained models from files
models = {}
for i in range(1, 9):
    models[f'L{i}'] = joblib.load(f'model_L{i}.joblib')  # Load each model

# Define your custom input data (ensure it has the same features used during training)
# Example test values: You can modify these as needed.
custom_input = {
    'IR': [0],     # Example current input (0A)
    'IY': [5],     # Example current input (5A)
    'IB': [10],    # Example current input (10A)
    'VR': [220],   # Example voltage input (220V)
    'VY': [225],   # Example voltage input (225V)
    'VB': [0],     # Example L input (0)
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame(custom_input)

# Initialize a dictionary to store predictions for each relay
predictions = {}

# Make predictions for each relay using the loaded models
for i in range(1, 9):
    # Predict the value for relay L{i}
    pred = models[f'L{i}'].predict(input_df)
    predictions[f'L{i}'] = pred[0]  # Store the prediction for relay L{i}

# Print the predicted values for each relay
for i in range(1, 9):
    print(f'Predicted value for L{i}: {predictions[f"L{i}"]}')
