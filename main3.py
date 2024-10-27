import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Load the dataset
data = pd.read_csv('high_quality_synthetic_data.csv')
X = data[['IR', 'IY', 'IB', 'VR', 'VY', 'VB']]
y = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Wrap the base model with MultiOutputClassifier for multi-output classification
multi_model = MultiOutputClassifier(base_model)

# Train the model on all relay outputs simultaneously
multi_model.fit(X_train, y_train)

# Make predictions on the test set
preds = multi_model.predict(X_test)

# Calculate accuracy for each relay
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], preds[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for L{i+1}: {accuracy}')

# Save the multi-output model
joblib.dump(multi_model, 'multi_output_model.joblib')

# Calculate and print average accuracy across all relays
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy}')
