"""
Dream11 AI — Optimizer
Integer Linear Programming team selector with full Dream11 constraints,
captain/VC 2x/1.5x multiplier baked into the objective.
"""
import logging
import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD
)
from config import TEAM_SIZE, BUDGET, MAX_FROM_TEAM, ROLE_LIMITS

logger = logging.getLogger(__name__)


def optimize_team(
    df: pd.DataFrame,
    captain: str | None = None,
    vice_captain: str | None = None,
    locked: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    """
    Selects an optimal 11-player Dream11 team.

    Parameters
    ----------
    df            : DataFrame with columns [player, team, role, credits, final_pred, ceiling, consistency]
    captain       : if given, force this player as captain
    vice_captain  : if given, force this player as VC
    locked        : list of player names forced into the team
    excluded      : list of player names excluded from the team
    """
    locked   = locked   or []
    excluded = excluded or []

    # ── Aggregate to one row per player ──────────────────────────────────────
    agg_dict = {"final_pred": "mean", "team": "first"}
    for col in ("credits", "role", "ceiling", "consistency"):
        if col in df.columns:
            agg_dict[col] = "first"
    df_a = df.groupby("player").agg(agg_dict).reset_index()

    # Defaults
    if "credits"     not in df_a.columns: df_a["credits"]     = 9.0
    if "role"        not in df_a.columns: df_a["role"]        = "BAT"
    if "ceiling"     not in df_a.columns: df_a["ceiling"]     = df_a["final_pred"] * 1.2
    if "consistency" not in df_a.columns: df_a["consistency"] = 1.0

    # Remove excluded
    df_a = df_a[~df_a["player"].isin(excluded)].reset_index(drop=True)
    players = df_a["player"].tolist()

    if len(players) < TEAM_SIZE:
        logger.warning(f"Only {len(players)} players available — returning all")
        return df_a

    # ── Risk-adjusted score ───────────────────────────────────────────────────
    # score = 0.6*pred + 0.2*ceiling + 0.2*consistency
    df_a["opt_score"] = (
        0.60 * df_a["final_pred"] +
        0.20 * df_a["ceiling"] +
        0.20 * df_a["consistency"].clip(0, 50)
    )

    # ── ILP ───────────────────────────────────────────────────────────────────
    x = {p: LpVariable(f"x_{i}", cat="Binary") for i, p in enumerate(players)}
    prob = LpProblem("Dream11", LpMaximize)
    score = dict(zip(df_a["player"], df_a["opt_score"]))
    credit = dict(zip(df_a["player"], df_a["credits"]))
    team_col = dict(zip(df_a["player"], df_a["team"]))
    role_col = dict(zip(df_a["player"], df_a["role"]))

    # Objective
    prob += lpSum(score[p] * x[p] for p in players)

    # Team size
    prob += lpSum(x[p] for p in players) == TEAM_SIZE

    # Budget
    prob += lpSum(credit[p] * x[p] for p in players) <= BUDGET

    # Max from one team
    for t in df_a["team"].unique():
        tp = [p for p in players if team_col[p] == t]
        prob += lpSum(x[p] for p in tp) <= MAX_FROM_TEAM

    # Role constraints (only apply if we have diverse role data)
    role_counts = df_a["role"].value_counts()
    has_diverse_roles = len(role_counts) >= 3  # at least WK, BAT, BOWL present

    if has_diverse_roles:
        for role, (lo, hi) in ROLE_LIMITS.items():
            rp = [p for p in players if role_col[p] == role]
            if rp:
                prob += lpSum(x[p] for p in rp) >= lo
                prob += lpSum(x[p] for p in rp) <= hi
    else:
        # Minimal fallback: just ensure at least one WK if present
        wk_players = [p for p in players if role_col[p] == "WK"]
        if wk_players:
            prob += lpSum(x[p] for p in wk_players) >= 1

    # Locked players
    for p in locked:
        if p in x:
            prob += x[p] == 1

    # Solve silently
    prob.solve(PULP_CBC_CMD(msg=0))

    selected = [p for p in players if x[p].varValue == 1]
    if len(selected) == 0:
        logger.warning("ILP found no solution — relaxing constraints")
        selected = df_a.nlargest(TEAM_SIZE, "opt_score")["player"].tolist()

    result = df_a[df_a["player"].isin(selected)].copy()

    # Captain / VC selection if not forced
    if captain and captain in result["player"].values:
        cap_row = captain
    else:
        cap_row = result.nlargest(1, "opt_score")["player"].iloc[0]

    if vice_captain and vice_captain in result["player"].values and vice_captain != cap_row:
        vc_row = vice_captain
    else:
        vc_row = result[result["player"] != cap_row].nlargest(1, "ceiling")["player"].iloc[0]

    result["role_tag"] = result["player"].apply(
        lambda p: "C" if p == cap_row else ("VC" if p == vc_row else "")
    )
    result["effective_pred"] = result.apply(
        lambda r: r["final_pred"] * 2.0 if r["role_tag"] == "C"
             else r["final_pred"] * 1.5 if r["role_tag"] == "VC"
             else r["final_pred"], axis=1
    )
    logger.info(f"Team selected: {len(result)} players, cap={cap_row}, vc={vc_row}")
    return result.reset_index(drop=True)