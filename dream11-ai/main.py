from src.preprocess import load_data, clean_data
from src.feature_engineering import create_player_features, create_ball_features, merge_features
from src.train_model import train_model
from src.dl_model import train_lstm
from src.predict import load_models, ensemble_predict
from src.optimizer import optimize_team

def run_pipeline():

    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)

    bat_feat = create_player_features(bat)
    ball_feat = create_ball_features(ball)

    df = merge_features(bat_feat, ball_feat)

    model = train_model(df)
    lstm_model = train_lstm(df)

    xgb, lstm = load_models()
    df = ensemble_predict(df, xgb, lstm)

    best_team = optimize_team(df)

    print("\n🔥 BEST DREAM11 TEAM:\n")
    print(best_team[["player", "final_pred"]])


if __name__ == "__main__":
    run_pipeline()