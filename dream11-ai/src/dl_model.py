"""
Dream11 AI — Deep Learning (LSTM + GRU)
Stacked LSTM with dropout, min-max normalisation, early stopping.
"""
import os
import numpy as np
import logging
import joblib

from config import MODEL_PATH, SEQ_LEN, RANDOM_SEED

logger = logging.getLogger(__name__)


# ─── Sequence builder ────────────────────────────────────────────────────────
def create_sequences(df, seq_len=SEQ_LEN):
    """Build (X, y) arrays from per-player match history."""
    X_list, y_list = [], []
    for player, grp in df.groupby("player"):
        grp  = grp.sort_values("match_id")
        pts  = grp["Batting_FP"].values.astype(np.float32)
        if len(pts) <= seq_len:
            continue
        for i in range(len(pts) - seq_len):
            X_list.append(pts[i:i+seq_len])
            y_list.append(pts[i+seq_len])
    if not X_list:
        return np.empty((0, seq_len, 1)), np.empty(0)
    X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ─── Model builder ────────────────────────────────────────────────────────────
def _build_lstm(seq_len: int):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        tf.random.set_seed(RANDOM_SEED)
        model = Sequential([
            Input(shape=(seq_len, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64,  return_sequences=True),
            Dropout(0.2),
            GRU(32),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mae", metrics=["mse"])
        return model
    except ImportError:
        logger.warning("TensorFlow not available — LSTM disabled")
        return None


# ─── Training ─────────────────────────────────────────────────────────────────
def train_lstm(df):
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        logger.warning("TensorFlow not installed; skipping LSTM training")
        return None

    os.makedirs(MODEL_PATH, exist_ok=True)
    X, y = create_sequences(df, SEQ_LEN)
    if len(X) == 0:
        logger.warning("Not enough sequences to train LSTM")
        return None

    # Normalise
    y_max = y.max() if y.max() > 0 else 1.0
    y_n   = y / y_max
    X_n   = X / y_max

    # Train / val split (last 10%)
    split = int(len(X_n) * 0.9)
    X_tr, X_va = X_n[:split], X_n[split:]
    y_tr, y_va = y_n[:split], y_n[split:]

    model = _build_lstm(SEQ_LEN)
    if model is None:
        return None

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=4, monitor="val_loss", verbose=0),
    ]
    model.fit(
        X_tr, y_tr,
        validation_data = (X_va, y_va),
        epochs          = 60,
        batch_size      = 64,
        callbacks       = callbacks,
        verbose         = 0,
    )
    model.save(os.path.join(MODEL_PATH, "lstm_model.keras"))
    joblib.dump(y_max, os.path.join(MODEL_PATH, "lstm_scale.pkl"))
    logger.info(f"LSTM trained on {len(X_tr):,} sequences, saved.")
    return model


# ─── Inference ────────────────────────────────────────────────────────────────
def lstm_predict(df, lstm_model, seq_len=SEQ_LEN):
    """
    Produces a per-player LSTM prediction using their last `seq_len` matches.
    Players with fewer than seq_len matches fall back to their avg_last10.
    """
    if lstm_model is None:
        df["lstm_pred"] = df.get("avg_last10", 0)
        return df

    y_max_path = os.path.join(MODEL_PATH, "lstm_scale.pkl")
    y_max = joblib.load(y_max_path) if os.path.exists(y_max_path) else 1.0

    pred_dict = {}
    for player, grp in df.groupby("player"):
        grp  = grp.sort_values("match_id")
        pts  = grp["Batting_FP"].values[-seq_len:].astype(np.float32)
        if len(pts) < seq_len:
            pred_dict[player] = float(grp["Batting_FP"].mean())
            continue
        seq = (pts / y_max).reshape(1, seq_len, 1)
        try:
            pred = float(lstm_model.predict(seq, verbose=0)[0][0]) * y_max
        except Exception:
            pred = float(grp["Batting_FP"].mean())
        pred_dict[player] = pred

    df["lstm_pred"] = df["player"].map(pred_dict).fillna(0)
    return df