"""
Dream11 AI — Captain / Vice-Captain selector
Blends safety (consistency, floor) with upside (ceiling) based on team strategy.
"""
import pandas as pd


def select_captain_vice(df: pd.DataFrame, strategy: str = "balanced"):
    """
    Returns (captain_name, vice_captain_name).

    strategy: 'safe' | 'balanced' | 'aggressive'
    """
    df = df.copy()

    if strategy == "safe":
        df["c_score"]  = 0.6*df["final_pred"] + 0.2*df["consistency"] + 0.2*df["floor"]
        df["vc_score"] = 0.7*df["final_pred"] + 0.1*df["consistency"] + 0.2*df["ceiling"]
    elif strategy == "aggressive":
        df["c_score"]  = 0.4*df["final_pred"] + 0.6*df["ceiling"]
        df["vc_score"] = 0.3*df["final_pred"] + 0.7*df["ceiling"]
    else:  # balanced
        df["c_score"]  = 0.5*df["final_pred"] + 0.3*df["ceiling"] + 0.2*df["consistency"]
        df["vc_score"] = 0.6*df["final_pred"] + 0.4*df["ceiling"]

    captain    = df.nlargest(1, "c_score")["player"].iloc[0]
    vice_df    = df[df["player"] != captain]
    vice_cap   = vice_df.nlargest(1, "vc_score")["player"].iloc[0] if len(vice_df) else captain

    return captain, vice_cap


def captain_stats(df: pd.DataFrame, captain: str, vice_captain: str) -> dict:
    """Return a summary dict for the C/VC choices."""
    cap_row = df[df["player"] == captain].iloc[0]
    vc_row  = df[df["player"] == vice_captain].iloc[0]
    return {
        "captain":      captain,
        "cap_pred":     round(float(cap_row["final_pred"]), 1),
        "cap_ceiling":  round(float(cap_row["ceiling"]),    1),
        "vice_captain": vice_captain,
        "vc_pred":      round(float(vc_row["final_pred"]),  1),
        "vc_ceiling":   round(float(vc_row["ceiling"]),     1),
    }