import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
from .config import MODEL_PATH
import os

SEQ_LEN = 10

def create_sequences(df):
    sequences = []
    targets = []

    for player, group in df.groupby("player"):
        group = group.sort_values("match_id")

        pts = group["Batting_FP"].values

        for i in range(len(pts) - SEQ_LEN):
            sequences.append(pts[i:i+SEQ_LEN])
            targets.append(pts[i+SEQ_LEN])

    return np.array(sequences), np.array(targets)


def train_lstm(df):
    X, y = create_sequences(df)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(os.path.join(MODEL_PATH, "lstm_model.keras"))
    return model