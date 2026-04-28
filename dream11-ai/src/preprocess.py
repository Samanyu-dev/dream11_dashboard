import pandas as pd
from .config import DATA_PATH

def load_data():
    ball = pd.read_csv(DATA_PATH + "all_matches_ball_by_ball1.csv")
    bat = pd.read_csv(DATA_PATH + "Batting_data.csv")
    match = pd.read_csv(DATA_PATH + "ipl_data.csv")
    return ball, bat, match


def clean_data(ball, bat, match):
    # Basic cleaning
    bat = bat.dropna(subset=["fullName"])
    ball = ball.dropna(subset=["batter", "bowler"])

    # Standardize names
    bat["player"] = bat["fullName"].str.strip()
    ball["player"] = ball["batter"].str.strip()

    # Add match_id if not present
    if "match_id" not in bat.columns:
        bat["match_id"] = bat.groupby("player").cumcount()

    # Create player to team mapping from ball data
    player_team = ball.groupby("player")["batting_team"].first().reset_index()
    player_team.columns = ["player", "team"]

    bat = bat.merge(player_team, on="player", how="left")

    return ball, bat, match