import os
from xgboost import XGBRegressor
import joblib
from .config import MODEL_PATH

def train_model(df):

    features = [
        "avg_last5",
        "avg_last10",
        "std_last10",
        "consistency",
        "boundary_ratio",
        "avg_runs_ball",
        "total_wickets"
    ]

    X = df[features].fillna(0)
    y = df["Batting_FP"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05
    )

    model.fit(X, y)

    os.makedirs(MODEL_PATH, exist_ok=True)

    path = os.path.join(MODEL_PATH, "xgb_model.pkl")
    print(f"Saving to: {path}")

    joblib.dump(model, path)

    return model