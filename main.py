import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load and preprocess the data
data = pd.read_csv('diverse_synthetic_data.csv')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

time_steps = 10
X = []
y = []

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, :-8])
    y.append(scaled_data[i, -8:])

X, y = np.array(X), np.array(y)

# Define the LSTM model with increased complexity
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=256))
model.add(Dropout(0.4))
model.add(Dense(8, activation='sigmoid'))

# Compile the model with a different optimizer
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping, model checkpoint, and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('complex_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model with more epochs and a larger batch size
model.fit(X, y, epochs=300, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(X)
predicted_relay_states = (predictions > 0.5).astype(int)
print("Predictions:\n", predicted_relay_states[:5])
