"""
Dream11 AI — Preprocessing
Loads and cleans raw CSV data, maps player→team, standardises names.
"""
import pandas as pd
import numpy as np
import logging
from config import BALL_CSV, BATTING_CSV, IPL_CSV

logger = logging.getLogger(__name__)


# ── Name normalisation helpers ────────────────────────────────────────────────
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("  ", " ")


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_data():
    logger.info("Loading raw CSVs …")
    ball  = pd.read_csv(BALL_CSV,    low_memory=False)
    bat   = pd.read_csv(BATTING_CSV, low_memory=False)
    match = pd.read_csv(IPL_CSV,     low_memory=False)
    logger.info(f"  ball={len(ball):,}  bat={len(bat):,}  match={len(match):,}")
    return ball, bat, match


def clean_data(ball: pd.DataFrame, bat: pd.DataFrame, match: pd.DataFrame):
    """Return cleaned (ball, bat, match) dataframes."""

    # ── Ball-by-ball ─────────────────────────────────────────────────────────
    ball = ball.dropna(subset=["batter", "bowler"]).copy()
    ball["batter"]  = ball["batter"].str.strip()
    ball["bowler"]  = ball["bowler"].str.strip()
    ball["boundary"]      = pd.to_numeric(ball["boundary"],      errors="coerce").fillna(0)
    ball["wicket_taken"]  = pd.to_numeric(ball["wicket_taken"],  errors="coerce").fillna(0).astype(int)
    ball["runs_batter"]   = pd.to_numeric(ball["runs_batter"],   errors="coerce").fillna(0)
    ball["pitch_type"]    = ball["pitch_type"].fillna("Unknown")
    ball["date"]          = pd.to_datetime(ball["date"], dayfirst=True, errors="coerce")

    # ── Batting ───────────────────────────────────────────────────────────────
    bat = bat.dropna(subset=["fullName"]).copy()
    bat["player"]     = bat["fullName"].str.strip()
    bat["season"]     = pd.to_numeric(bat["season"], errors="coerce").fillna(0).astype(int)
    bat["Batting_FP"] = pd.to_numeric(bat["Batting_FP"], errors="coerce").fillna(0)
    bat["runs"]        = pd.to_numeric(bat["runs"],        errors="coerce").fillna(0)
    bat["balls"]       = pd.to_numeric(bat["balls"],       errors="coerce").fillna(0)
    bat["fours"]       = pd.to_numeric(bat["fours"],       errors="coerce").fillna(0)
    bat["sixes"]       = pd.to_numeric(bat["sixes"],       errors="coerce").fillna(0)
    bat["strike_rate"] = pd.to_numeric(bat["strike_rate"], errors="coerce").fillna(0)

    # Assign match_id if missing
    if "match_id" not in bat.columns or bat["match_id"].isna().all():
        bat["match_id"] = bat.groupby("player").cumcount()

    # Add team column from ball data (most frequent batting_team per batter)
    player_team = (
        ball.groupby("batter")["batting_team"]
        .agg(lambda x: x.mode()[0] if len(x) else "Unknown")
        .reset_index()
        .rename(columns={"batter": "player", "batting_team": "team"})
    )
    bat = bat.merge(player_team, on="player", how="left")
    bat["team"] = bat["team"].fillna(
        bat.get("batting_team", bat.get("home_team", "Unknown"))
    )

    # ── Match summary ─────────────────────────────────────────────────────────
    match = match.copy()
    match["date"] = pd.to_datetime(match["date"], errors="coerce")

    logger.info(f"Clean: ball={len(ball):,}  bat={len(bat):,}")
    return ball, bat, match


def filter_teams(bat: pd.DataFrame, ball: pd.DataFrame, team1: str, team2: str):
    """Restrict dataframes to players of two specific teams."""
    bat_f  = bat[bat["team"].isin([team1, team2])].copy()
    ball_f = ball[ball["batting_team"].isin([team1, team2])].copy()
    return bat_f, ball_f