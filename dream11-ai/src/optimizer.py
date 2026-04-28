from pulp import *

def optimize_team(df):

    # Aggregate to one row per player
    agg_dict = {"final_pred": "mean"}
    if "credits" in df.columns:
        agg_dict["credits"] = "first"
    if "role" in df.columns:
        agg_dict["role"] = "first"
    if "team" in df.columns:
        agg_dict["team"] = "first"

    df_agg = df.groupby("player").agg(agg_dict).reset_index()
    df_agg = df_agg.set_index("player")

    players = df_agg.index.tolist()

    x = {p: LpVariable(p, cat='Binary') for p in players}

    prob = LpProblem("Dream11", LpMaximize)

    # Objective
    prob += lpSum(df_agg.loc[p, "final_pred"] * x[p] for p in players)

    # Team size
    prob += lpSum(x[p] for p in players) == 11

    # Budget constraint
    if "credits" in df_agg.columns:
        prob += lpSum(df_agg.loc[p, "credits"] * x[p] for p in players) <= 100
    else:
        prob += lpSum(9 * x[p] for p in players) <= 100

    # ROLE CONSTRAINTS
    if "role" in df_agg.columns:
        prob += lpSum(x[p] for p in players if df_agg.loc[p, "role"] == "WK") >= 1
        prob += lpSum(x[p] for p in players if df_agg.loc[p, "role"] == "BAT") >= 3
        prob += lpSum(x[p] for p in players if df_agg.loc[p, "role"] == "AR") >= 1
        prob += lpSum(x[p] for p in players if df_agg.loc[p, "role"] == "BOWL") >= 3

    # TEAM BALANCE
    if "team" in df_agg.columns:
        teams = df_agg["team"].unique()
        for t in teams:
            prob += lpSum(x[p] for p in players if df_agg.loc[p, "team"] == t) <= 7

    prob.solve()

    selected = [p for p in players if x[p].value() == 1]

    return df_agg.loc[selected].reset_index()