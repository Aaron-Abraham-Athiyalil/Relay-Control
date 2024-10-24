import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras_tuner import RandomSearch
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Ensure TensorFlow is using optimal CPU instructions if possible
print(f"TensorFlow version: {tf.__version__}")

# Load your data (replace this with actual data loading code)
# Assume 'data' is your features and 'labels' is your target variable
# For now, let's use dummy data for illustration
data = np.random.rand(1000, 10)  # 1000 samples, 10 features
labels = np.random.randint(2, size=1000)  # Binary classification

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Verify the shape of the data
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Define a simple LSTM model for tuning
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                   input_shape=(X_train.shape[1], 1)))  # Assuming X_train has shape (samples, timesteps, features)
    model.add(Dense(1, activation='sigmoid'))  # For binary classification

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Reshape data if necessary (LSTM expects 3D input: samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Set up early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

# Save the best model during training
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1)

# Initialize Keras Tuner with Random Search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='lstm_tuning'
)

# Begin search for the best hyperparameters
tuner.search(X_train, y_train, epochs=500, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
