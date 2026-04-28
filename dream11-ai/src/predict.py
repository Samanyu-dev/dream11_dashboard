"""
Dream11 AI — Prediction
Loads trained models and produces ensemble fantasy-point predictions.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from config import MODEL_PATH, W_XGB, W_LSTM, W_RF
from dl_model import lstm_predict

logger = logging.getLogger(__name__)


# ─── Model loading ───────────────────────────────────────────────────────────
def load_models():
    paths = {
        "xgb":    os.path.join(MODEL_PATH, "xgb_model.pkl"),
        "lgb":    os.path.join(MODEL_PATH, "lgb_model.pkl"),
        "rf":     os.path.join(MODEL_PATH, "rf_model.pkl"),
        "scaler": os.path.join(MODEL_PATH, "scaler.pkl"),
        "feats":  os.path.join(MODEL_PATH, "feature_cols.json"),
    }
    models = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            logger.warning(f"Model file missing: {path}")
            models[name] = None
        else:
            if name == "feats":
                with open(path) as f:
                    models[name] = json.load(f)
            else:
                models[name] = joblib.load(path)

    # LSTM is optional (needs TF)
    lstm_path = os.path.join(MODEL_PATH, "lstm_model.keras")
    try:
        from tensorflow.keras.models import load_model as tf_load
        models["lstm"] = tf_load(lstm_path) if os.path.exists(lstm_path) else None
    except Exception:
        models["lstm"] = None

    return models


# ─── Ensemble inference ──────────────────────────────────────────────────────
def ensemble_predict(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Adds prediction columns to df:
      xgb_pred, lgb_pred, rf_pred, lstm_pred, final_pred
    """
    feats  = models.get("feats") or []
    scaler = models.get("scaler")

    # Build feature matrix (only use columns that exist)
    avail = [f for f in feats if f in df.columns]
    if not avail:
        logger.warning("No feature columns found — using avg_last10 as fallback")
        df["final_pred"] = df.get("avg_last10", pd.Series(0, index=df.index))
        return df

    X = df[avail].fillna(0).values
    if scaler is not None:
        X = scaler.transform(X)

    # Individual model predictions
    weight_sum = 0.0
    df["final_pred"] = 0.0

    for key, w, col in [("xgb", W_XGB, "xgb_pred"), ("lgb", W_RF * 0.6, "lgb_pred"), ("rf", W_RF * 0.4, "rf_pred")]:
        m = models.get(key)
        if m is not None:
            try:
                df[col] = m.predict(X)
                df["final_pred"] += w * df[col]
                weight_sum += w
            except Exception as e:
                logger.warning(f"{key} prediction failed: {e}")
                df[col] = 0

    # LSTM
    if models.get("lstm") is not None:
        df = lstm_predict(df, models["lstm"])
        df["final_pred"] += W_LSTM * df["lstm_pred"]
        weight_sum += W_LSTM
    else:
        df["lstm_pred"] = 0

    # Normalise by actual weights used
    if weight_sum > 0:
        df["final_pred"] /= weight_sum
        df["final_pred"]  *= (W_XGB + W_LSTM + W_RF)

    df["final_pred"] = df["final_pred"].clip(lower=0)
    logger.info(f"Predictions: mean={df['final_pred'].mean():.2f}  max={df['final_pred'].max():.2f}")
    return df


# ─── Convenience: aggregate to one row per player ────────────────────────────
def _infer_role(row) -> str:
    """Infer player role from their bowling/batting stats."""
    wkts  = float(row.get("total_wickets", 0) or 0)
    avg10 = float(row.get("avg_last10",    0) or 0)
    eco   = float(row.get("economy",       0) or 0)
    # Simple heuristic: high wickets + economy → bowler; mix → allrounder
    if wkts > 20 and avg10 < 25:
        return "BOWL"
    if wkts > 10 and avg10 > 20:
        return "AR"
    return "BAT"


def aggregate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses multi-row player dataframe to one row per player
    with mean of predictions and last-known metadata.
    """
    agg_cols = {
        "final_pred":  ("final_pred",  "mean"),
        "avg_last10":  ("avg_last10",  "last"),
        "std_last10":  ("std_last10",  "last"),
        "ceiling":     ("ceiling",     "last"),
        "floor":       ("floor",       "last"),
        "consistency": ("consistency", "last"),
        "team":        ("team",        "last"),
    }
    for col in ("credits", "role", "total_wickets", "economy"):
        if col in df.columns:
            agg_cols[col] = (col, "last")

    named_agg = {k: pd.NamedAgg(column=v[0], aggfunc=v[1]) for k, v in agg_cols.items()}
    agg = df.groupby("player").agg(**named_agg).reset_index()

    # Default credits
    if "credits" not in df.columns:
        agg["credits"] = 9.0

    # Infer roles from stats if all same or missing
    has_role = "role" in df.columns
    if not has_role or df["role"].nunique() <= 1:
        # Bring in bowling stats for inference
        for col in ("total_wickets", "economy", "avg_last10"):
            if col not in agg.columns and col in df.columns:
                agg[col] = df.groupby("player")[col].last().values
        agg["role"] = agg.apply(_infer_role, axis=1)

    # Ensure at least 1 WK per squad (assign to highest-consistency BAT)
    if "WK" not in agg["role"].values:
        idx = agg[agg["role"] == "BAT"].nlargest(1, "consistency").index
        if len(idx):
            agg.loc[idx[0], "role"] = "WK"

    return agg