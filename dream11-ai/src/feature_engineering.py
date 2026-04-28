import pandas as pd
import numpy as np

def create_player_features(bat):
    bat = bat.sort_values(["player", "match_id"])

    bat["avg_last5"] = bat.groupby("player")["Batting_FP"].rolling(5).mean().reset_index(0, drop=True)
    bat["avg_last10"] = bat.groupby("player")["Batting_FP"].rolling(10).mean().reset_index(0, drop=True)

    bat["std_last10"] = bat.groupby("player")["Batting_FP"].rolling(10).std().reset_index(0, drop=True)

    bat["consistency"] = 1 / (bat["std_last10"] + 1)

    bat["boundary_ratio"] = (bat["fours"] + bat["sixes"]) / bat["balls"]

    return bat


def create_ball_features(ball):
    agg = ball.groupby("player").agg({
        "runs_batter": "mean",
        "wicket_taken": "sum"
    }).reset_index()

    agg.columns = ["player", "avg_runs_ball", "total_wickets"]

    return agg


def merge_features(bat, ball_feat):
    df = bat.merge(ball_feat, on="player", how="left")
    df = df.fillna(0)
    return df