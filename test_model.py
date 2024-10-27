import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the saved multi-output model
multi_model = joblib.load('multi_output_model.joblib')

# Load the test dataset 
test_data = pd.read_csv('high_quality_synthetic_data.csv')  # Ensure this file has similar columns to the training data
X_test = test_data[['IR', 'IY', 'IB', 'VR', 'VY', 'VB']]
y_test = test_data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']]

# Make predictions using the loaded model
preds = multi_model.predict(X_test)# Make predictions using the loaded model
preds = multi_model.predict(X_test)

# Print only even values for each relay
for i in range(y_test.shape[1]):
    even_preds = [pred for pred in preds[:, i] if pred % 2 == 0]  # Filter even values
    print(f'Even values for L{i+1}: {even_preds}')

# Calculate accuracy for each relay
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], preds[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for L{i+1}: {accuracy}')

# Calculate and print average accuracy across all relays
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy on New Test Data: {average_accuracy}')
