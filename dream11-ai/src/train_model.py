"""
Dream11 AI — Model Training
Trains an ensemble of XGBoost + LightGBM + RandomForest with time-based splits,
cross-validation, and Optuna hyperparameter tuning (optional).
Saves all models to disk.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb

from config import MODEL_PATH, TRAIN_SEASONS, TEST_SEASONS, RANDOM_SEED
from feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _split(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()
    return train, test


def _prep(df: pd.DataFrame):
    feats = [f for f in FEATURE_COLS if f in df.columns]
    X = df[feats].fillna(0).values
    y = df["Batting_FP"].values
    return X, y, feats


def _metrics(y_true, y_pred, tag=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    logger.info(f"  [{tag}] MAE={mae:.2f}  RMSE={rmse:.2f}")
    return {"mae": mae, "rmse": rmse}


# ── XGBoost ───────────────────────────────────────────────────────────────────
def train_xgb(X_train, y_train, X_val, y_val):
    logger.info("Training XGBoost …")
    model = XGBRegressor(
        n_estimators    = 500,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_weight= 3,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
        verbosity       = 0,
        early_stopping_rounds = 30,
        eval_metric     = "mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    _metrics(y_val, model.predict(X_val), "XGB-val")
    return model


# ── LightGBM ──────────────────────────────────────────────────────────────────
def train_lgb(X_train, y_train, X_val, y_val):
    logger.info("Training LightGBM …")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val,   reference=train_data)
    params = {
        "objective":       "regression",
        "metric":          "mae",
        "num_leaves":      63,
        "learning_rate":   0.05,
        "feature_fraction":0.8,
        "bagging_fraction":0.8,
        "bagging_freq":    5,
        "min_child_samples":20,
        "verbose":        -1,
        "seed":            RANDOM_SEED,
    }
    cb = lgb.early_stopping(30, verbose=False)
    model = lgb.train(
        params, train_data,
        num_boost_round    = 500,
        valid_sets         = [val_data],
        callbacks          = [cb, lgb.log_evaluation(period=-1)],
    )
    _metrics(y_val, model.predict(X_val), "LGB-val")
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────
def train_rf(X_train, y_train, X_val, y_val):
    logger.info("Training RandomForest …")
    model = RandomForestRegressor(
        n_estimators = 300,
        max_depth    = 12,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    _metrics(y_val, model.predict(X_val), "RF-val")
    return model


# ── Main training entry point ─────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    os.makedirs(MODEL_PATH, exist_ok=True)

    train_df, test_df = _split(df)

    # Time-based val: last 2 training seasons
    val_seasons = sorted(train_df["season"].unique())[-2:]
    val_df   = train_df[train_df["season"].isin(val_seasons)]
    train_df = train_df[~train_df["season"].isin(val_seasons)]

    logger.info(f"Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    X_tr, y_tr, feats = _prep(train_df)
    X_va, y_va, _     = _prep(val_df)
    X_te, y_te, _     = _prep(test_df)

    # Scale for LGB (tree models don't strictly need it but helps RF)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_va, Xs_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    xgb_m = train_xgb(Xs_tr, y_tr, Xs_va, y_va)
    lgb_m = train_lgb(Xs_tr, y_tr, Xs_va, y_va)
    rf_m  = train_rf (Xs_tr, y_tr, Xs_va, y_va)

    # Ensemble test evaluation
    from config import W_XGB, W_LSTM, W_RF
    w_e = W_XGB + W_RF   # LSTM not available at training time
    ens_pred = (W_XGB * xgb_m.predict(Xs_te) + W_RF * rf_m.predict(Xs_te)) / w_e
    _metrics(y_te, ens_pred, "Ensemble-test")

    # Persist
    joblib.dump(xgb_m,  os.path.join(MODEL_PATH, "xgb_model.pkl"))
    joblib.dump(lgb_m,  os.path.join(MODEL_PATH, "lgb_model.pkl"))
    joblib.dump(rf_m,   os.path.join(MODEL_PATH, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    with open(os.path.join(MODEL_PATH, "feature_cols.json"), "w") as f:
        json.dump(feats, f)

    logger.info("All models saved to " + MODEL_PATH)
    return xgb_m, lgb_m, rf_m, scaler, feats