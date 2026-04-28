"""
Dream11 AI — Grand League Strategy Engine
Generates N diverse teams across the safe→aggressive spectrum.
Uses ownership simulation + differential scoring + noise injection.
"""
import numpy as np
import pandas as pd
import logging
from optimizer import optimize_team
from captain import select_captain_vice, captain_stats
from config import N_GL_TEAMS, RANDOM_SEED

logger = logging.getLogger(__name__)
rng = np.random.default_rng(RANDOM_SEED)


def simulate_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate public ownership % based on predicted points rank."""
    df = df.copy()
    rank = df["final_pred"].rank(ascending=False)
    df["ownership"] = np.exp(-0.15 * rank) / np.exp(-0.15 * rank).sum()
    df["ownership"] = (df["ownership"] * 100).round(1)
    return df


def generate_gl_teams(df: pd.DataFrame, n_teams: int = N_GL_TEAMS) -> list[dict]:
    """
    Returns a list of n_teams dicts, each with:
      players, captain, vice_captain, strategy, predicted_score
    """
    df = simulate_ownership(df)
    df["differential_score"] = (1 - df["ownership"] / 100) * df["ceiling"]

    teams = []
    strategies = (
        ["safe"] +
        ["balanced"] * max(1, n_teams // 3) +
        ["aggressive"] * (n_teams - 1 - max(1, n_teams // 3))
    )
    if len(strategies) < n_teams:
        strategies += ["aggressive"] * (n_teams - len(strategies))

    locked_pools = _build_locked_pools(df, n_teams)

    for i in range(n_teams):
        strat = strategies[i]
        temp  = df.copy()

        if strat == "safe":
            temp["score"] = temp["final_pred"]
        elif strat == "balanced":
            temp["score"] = 0.7 * temp["final_pred"] + 0.3 * temp["consistency"]
        else:
            temp["score"] = (
                0.45 * temp["final_pred"] +
                0.35 * temp["differential_score"] +
                0.20 * temp["ceiling"]
            )
            # Inject diversity noise
            temp["score"] += rng.normal(0, 5, len(temp))

        temp["final_pred"] = temp["score"]

        locked = locked_pools[i] if i < len(locked_pools) else []

        try:
            team_df = optimize_team(temp, locked=locked)
            cap, vc = select_captain_vice(team_df, strategy=strat)
            c_stats = captain_stats(team_df, cap, vc)

            teams.append({
                "team_no":         i + 1,
                "strategy":        strat,
                "players":         team_df["player"].tolist(),
                "captain":         cap,
                "vice_captain":    vc,
                "captain_stats":   c_stats,
                "predicted_score": round(float(team_df["effective_pred"].sum()), 1) if "effective_pred" in team_df else 0,
                "differentials":   _get_differentials(team_df, df, top_n=3),
                "team_detail":     team_df[["player", "team", "role", "final_pred", "ownership", "role_tag"]].to_dict("records"),
            })
        except Exception as e:
            logger.error(f"Team {i+1} failed: {e}")

    logger.info(f"Generated {len(teams)} GL teams")
    return teams


def _build_locked_pools(df: pd.DataFrame, n: int) -> list[list]:
    """
    Each team locks 1-2 high-ceiling differentials to force variety.
    """
    diff_players = df.nlargest(min(8, len(df)), "differential_score")["player"].tolist()
    pools = []
    for i in range(n):
        if i == 0:
            pools.append([])  # safe team — no locks
        else:
            pick = min(2, len(diff_players))
            chosen = list(rng.choice(diff_players, pick, replace=False))
            pools.append(chosen)
    return pools


def _get_differentials(team_df: pd.DataFrame, full_df: pd.DataFrame, top_n=3) -> list:
    """Players in this team with low ownership (high differential)."""
    merged = team_df.merge(full_df[["player","ownership"]], on="player", how="left")
    diffs = merged.nsmallest(top_n, "ownership_y")["player"].tolist()
    return diffs