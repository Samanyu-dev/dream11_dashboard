import sys
from src.preprocess import load_data, clean_data
from src.feature_engineering import create_player_features, create_ball_features, merge_features
from src.predict import load_models, ensemble_predict
from src.optimizer import optimize_team

def generate_team_for_match(team1, team2):
    print(f"Generating Dream11 team for {team1} vs {team2}")

    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)

    # Filter players to only those from the two teams
    bat = bat[bat["team"].isin([team1, team2])]

    if bat.empty:
        print(f"No players found for teams {team1} and {team2}")
        return

    bat_feat = create_player_features(bat)
    ball_feat = create_ball_features(ball)
    ball_feat = ball_feat[ball_feat["player"].isin(bat["player"])]  # Filter ball features too

    df = merge_features(bat_feat, ball_feat)

    xgb, lstm = load_models()
    df = ensemble_predict(df, xgb, lstm)

    # Add GL features
    df["ceiling"] = df["avg_last10"] + df["std_last10"]
    df["risk"] = df["std_last10"]
    df["ownership"] = df["final_pred"].rank(pct=True)
    df["differential_score"] = (1 - df["ownership"]) * df["ceiling"]

    best_team = optimize_team(df)

    print("\n🔥 BEST DREAM11 TEAM:\n")
    print(best_team[["player", "team", "final_pred"]])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_team.py <team1> <team2>")
        print("Example: python generate_team.py CSK GT")
        sys.exit(1)

    team1 = sys.argv[1].title()
    team2 = sys.argv[2].title()

    generate_team_for_match(team1, team2)