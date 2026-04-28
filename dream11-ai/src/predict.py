import joblib
import numpy as np
from tensorflow.keras.models import load_model
from .config import MODEL_PATH
import os

def load_models():
    xgb = joblib.load(os.path.join(MODEL_PATH, "xgb_model.pkl"))
    lstm = load_model(os.path.join(MODEL_PATH, "lstm_model.keras"))
    return xgb, lstm


def lstm_predict(df, lstm):
    pred_dict = {}

    for player, group in df.groupby("player"):
        group = group.sort_values("match_id")

        pts = group["Batting_FP"].values[-10:]

        if len(pts) < 10:
            pred = 0
        else:
            seq = np.array(pts).reshape(1, 10, 1)
            pred = lstm.predict(seq, verbose=0)[0][0]

        pred_dict[player] = pred

    df["lstm_pred"] = df["player"].map(pred_dict)
    return df


def ensemble_predict(df, xgb, lstm):
    features = [
        "avg_last5", "avg_last10", "std_last10",
        "consistency", "boundary_ratio",
        "avg_runs_ball", "total_wickets"
    ]

    df["xgb_pred"] = xgb.predict(df[features].fillna(0))

    df = lstm_predict(df, lstm)

    df["final_pred"] = (
        0.6 * df["xgb_pred"] +
        0.4 * df["lstm_pred"]
    )

    return df