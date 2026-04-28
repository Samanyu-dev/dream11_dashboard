"""
Dream11 AI — Feature Engineering
Builds rich per-player, per-match features for the ML pipeline.
"""
import pandas as pd
import numpy as np
import logging
from config import ROLLING_WINDOWS

logger = logging.getLogger(__name__)


# ── Helper: safe rolling mean / std ──────────────────────────────────────────
def _rolling(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    rolled = series.rolling(window, min_periods=1)
    return getattr(rolled, func)().shift(1)   # shift(1) → no leakage


def create_player_features(bat: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player batting features from the Batting_data.csv.
    Returns a per-player-per-match dataframe.
    """
    bat = bat.sort_values(["player", "match_id"]).copy()

    grp = bat.groupby("player")

    # ── Rolling fantasy-point windows ────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        bat[f"avg_last{w}"]  = grp["Batting_FP"].transform(lambda s: _rolling(s, w, "mean"))
        bat[f"std_last{w}"]  = grp["Batting_FP"].transform(lambda s: _rolling(s, w, "std"))

    # Fill NaNs from early rows with career mean
    fp_cols = [c for c in bat.columns if c.startswith(("avg_last", "std_last"))]
    for col in fp_cols:
        bat[col] = bat[col].fillna(grp["Batting_FP"].transform("mean"))
    bat[fp_cols] = bat[fp_cols].fillna(0)

    # ── Consistency score ─────────────────────────────────────────────────────
    bat["consistency"]     = 1 / (bat["std_last10"] + 1)

    # ── Boundary rate ─────────────────────────────────────────────────────────
    safe_balls = bat["balls"].replace(0, np.nan)
    bat["boundary_ratio"]  = (bat["fours"] + bat["sixes"]) / safe_balls
    bat["six_ratio"]       = bat["sixes"] / safe_balls
    bat["boundary_ratio"]  = bat["boundary_ratio"].fillna(0).clip(0, 1)
    bat["six_ratio"]       = bat["six_ratio"].fillna(0).clip(0, 1)

    # ── Strike rate (normalised) ──────────────────────────────────────────────
    bat["sr_norm"] = bat["strike_rate"] / 150.0

    # ── Batting position importance weight ───────────────────────────────────
    pos_map = {1: 1.2, 2: 1.2, 3: 1.15, 4: 1.1, 5: 1.05, 6: 1.0, 7: 0.9}
    bat["pos_weight"] = bat["batting_position"].map(pos_map).fillna(0.85)

    # ── Venue performance ─────────────────────────────────────────────────────
    venue_avg = (
        bat.groupby(["player", "venue"])["Batting_FP"]
        .transform("mean")
        .rename("venue_avg_fp")
    )
    bat["venue_avg_fp"] = venue_avg.fillna(bat["avg_last10"])

    # ── Career aggregates (shifted to avoid leakage) ──────────────────────────
    bat["career_avg"]    = grp["Batting_FP"].transform(lambda s: s.expanding().mean().shift(1)).fillna(0)
    bat["career_max"]    = grp["Batting_FP"].transform(lambda s: s.expanding().max().shift(1)).fillna(0)
    bat["ceiling"]       = bat["avg_last10"] + bat["std_last10"]
    bat["floor"]         = (bat["avg_last10"] - bat["std_last10"]).clip(lower=0)

    # ── Opponent strength (avg FP conceded to this batting_team by bowling_team) ─
    opp_avg = (
        bat.groupby(["bowling_team"])["Batting_FP"]
        .transform("mean")
        .rename("opp_strength")
    )
    bat["opp_strength"] = opp_avg.fillna(bat["avg_last10"])

    logger.info(f"Player features: {bat.shape[1]} cols, {len(bat):,} rows")
    return bat


def create_ball_features(ball: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player bowling + batting aggregates from ball-by-ball data.
    """
    # ── Batting aggregates from ball data ─────────────────────────────────────
    bat_agg = ball.groupby("batter").agg(
        avg_runs_ball  = ("runs_batter", "mean"),
        total_runs_bbb = ("runs_batter", "sum"),
        fours_bbb      = ("boundary", lambda x: (x == 4).sum()),
        sixes_bbb      = ("boundary", lambda x: (x == 6).sum()),
        balls_faced    = ("runs_batter", "count"),
        dot_balls      = ("runs_batter", lambda x: (x == 0).sum()),
    ).reset_index().rename(columns={"batter": "player"})

    bat_agg["dot_pct"] = bat_agg["dot_balls"] / bat_agg["balls_faced"].replace(0, np.nan)
    bat_agg["dot_pct"] = bat_agg["dot_pct"].fillna(0)

    # ── Bowling aggregates from ball data ─────────────────────────────────────
    bowl_agg = ball.groupby("bowler").agg(
        total_wickets   = ("wicket_taken", "sum"),
        balls_bowled    = ("bowler",       "count"),
        runs_conceded   = ("runs_batter",  "sum"),
    ).reset_index().rename(columns={"bowler": "player"})

    bowl_agg["economy"]       = (bowl_agg["runs_conceded"] / bowl_agg["balls_bowled"] * 6).replace([np.inf, np.nan], 12)
    bowl_agg["wickets_per10"] = (bowl_agg["total_wickets"]  / bowl_agg["balls_bowled"] * 10).replace([np.inf, np.nan], 0)

    # ── Pitch type performance ────────────────────────────────────────────────
    pitch_agg = ball.groupby(["batter", "pitch_type"])["runs_batter"].mean().unstack(fill_value=0)
    pitch_agg.columns = [f"avg_runs_{c.lower().replace('-','_').replace(' ','_')}" for c in pitch_agg.columns]
    pitch_agg = pitch_agg.reset_index().rename(columns={"batter": "player"})

    # ── Merge all ─────────────────────────────────────────────────────────────
    out = bat_agg.merge(bowl_agg, on="player", how="outer")
    out = out.merge(pitch_agg,   on="player", how="left")
    out = out.fillna(0)

    logger.info(f"Ball features: {out.shape[1]} cols, {len(out):,} rows")
    return out


def merge_features(bat: pd.DataFrame, ball_feat: pd.DataFrame) -> pd.DataFrame:
    """Merge batting and ball-derived features into one training dataframe."""
    df = bat.merge(ball_feat, on="player", how="left")
    df = df.fillna(0)
    # Final sanity: remove rows with no target
    df = df[df["Batting_FP"].notna()].copy()
    logger.info(f"Merged features: {df.shape[1]} cols, {len(df):,} rows")
    return df


# ── Feature column list ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "avg_last5", "avg_last10", "avg_last20",
    "std_last5", "std_last10", "std_last20",
    "consistency", "boundary_ratio", "six_ratio", "sr_norm",
    "pos_weight", "venue_avg_fp", "career_avg", "career_max",
    "ceiling", "floor", "opp_strength",
    "avg_runs_ball", "dot_pct",
    "total_wickets", "economy", "wickets_per10",
]