"""
Dream11 AI — Insights Engine
Generates human-readable insights from prediction + feature data.
"""
import pandas as pd


def generate_insights(df: pd.DataFrame, match_context: dict | None = None) -> list[str]:
    """
    Returns a list of insight strings about player form, risk, and matchup.

    df must have: player, final_pred, avg_last10, std_last10, ceiling,
                  consistency, ownership, venue_avg_fp (optional),
                  opp_strength (optional)
    """
    insights = []
    df = df.copy()

    # Ensure required columns exist
    for col, default in [("ownership", 50), ("venue_avg_fp", df.get("avg_last10", pd.Series(0))),
                         ("opp_strength", df.get("avg_last10", pd.Series(0)))]:
        if col not in df.columns:
            df[col] = default

    # ── Form insights ────────────────────────────────────────────────────────
    hot_threshold   = df["avg_last10"].quantile(0.80)
    cold_threshold  = df["avg_last10"].quantile(0.25)

    for _, row in df.iterrows():
        p = row["player"]

        if row.get("avg_last10", 0) >= hot_threshold:
            insights.append(f"🔥 {p} is in top form — averaging {row['avg_last10']:.0f} pts in last 10 matches")

        if row.get("avg_last10", 99) <= cold_threshold and row.get("avg_last10", 0) > 0:
            insights.append(f"❄️ {p} is struggling — low recent average ({row['avg_last10']:.0f} pts)")

        # High ceiling, high variance → GL pick
        if row.get("ceiling", 0) > 80 and row.get("std_last10", 0) > 25:
            insights.append(f"⚡ {p} is high risk–high reward: ceiling {row['ceiling']:.0f}, volatility σ={row['std_last10']:.0f}")

        # Safe pick
        if row.get("consistency", 0) > 1.5 and row.get("final_pred", 0) > 40:
            insights.append(f"🛡️ {p} is a consistent, safe pick — reliability score {row['consistency']:.2f}")

        # Differential
        if row.get("ownership", 100) < 20 and row.get("final_pred", 0) > 50:
            insights.append(f"💎 {p} is an under-owned differential ({row['ownership']:.0f}% ownership) with strong upside")

        # Venue form
        if "venue_avg_fp" in df.columns:
            if row.get("venue_avg_fp", 0) > row.get("avg_last10", 0) * 1.2:
                insights.append(f"🏟️ {p} historically excels at this venue ({row['venue_avg_fp']:.0f} avg pts)")

    # ── Overall match insights ────────────────────────────────────────────────
    top_pick = df.nlargest(1, "final_pred")["player"].iloc[0]
    insights.insert(0, f"🏆 Top predicted performer: {top_pick} ({df[df['player']==top_pick]['final_pred'].values[0]:.0f} pts)")

    top_diff = df.nsmallest(1, "ownership").nlargest(1, "ceiling")
    if len(top_diff):
        insights.insert(1, f"💡 Best differential: {top_diff['player'].values[0]} (ceiling {top_diff['ceiling'].values[0]:.0f} pts, {top_diff['ownership'].values[0]:.0f}% owned)")

    return insights[:15]  # cap at 15 insights


def player_report(df: pd.DataFrame, player: str) -> dict:
    """Return a detailed dict report for a single player."""
    row = df[df["player"] == player]
    if row.empty:
        return {"error": f"Player '{player}' not found"}
    r = row.iloc[0]
    return {
        "player":        player,
        "predicted_pts": round(float(r.get("final_pred", 0)),   1),
        "avg_last5":     round(float(r.get("avg_last5",  0)),   1),
        "avg_last10":    round(float(r.get("avg_last10", 0)),   1),
        "ceiling":       round(float(r.get("ceiling",    0)),   1),
        "floor":         round(float(r.get("floor",      0)),   1),
        "consistency":   round(float(r.get("consistency",0)),   2),
        "std_last10":    round(float(r.get("std_last10", 0)),   1),
        "ownership_pct": round(float(r.get("ownership",  0)),   1),
        "team":          str(r.get("team", "?")),
        "role":          str(r.get("role", "?")),
    }