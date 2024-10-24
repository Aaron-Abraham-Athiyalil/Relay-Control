import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch

import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load and preprocess the data
data = pd.read_csv('high_quality_synthetic_data.csv')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

time_steps = 10
X = []
y = []

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, :-8])
    y.append(scaled_data[i, -8:])

X, y = np.array(X), np.array(y)

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=128, max_value=512, step=32)))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(8, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='lstm_tuning'
)

tuner.search_space_summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('optimized_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

tuner.search(X, y, epochs=300, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy = best_model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

predictions = best_model.predict(X)
predicted_relay_states = (predictions > 0.5).astype(int)
print("Predictions:\n", predicted_relay_states[:5])
