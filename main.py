import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load and preprocess the data
data = pd.read_csv('balanced_synthetic_data.csv')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

time_steps = 10
X = []
y = []

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, :-8])
    y.append(scaled_data[i, -8:])

X, y = np.array(X), np.array(y)

# Simplify the model for initial debugging
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dense(8, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('simple_model.keras', save_best_only=True)

model.fit(X, y, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

predictions = model.predict(X)
predicted_relay_states = (predictions > 0.5).astype(int)
print("Predictions:\n", predicted_relay_states[:5])
