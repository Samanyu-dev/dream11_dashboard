def select_captain_vice(df):

    df = df.copy()

    # Score = blend of safety + upside
    df["c_score"] = (
        0.5 * df["final_pred"] +
        0.3 * df["ceiling"] +
        0.2 * df["consistency"]
    )

    df["vc_score"] = (
        0.6 * df["final_pred"] +
        0.4 * df["ceiling"]
    )

    captain = df.sort_values("c_score", ascending=False).iloc[0]["player"]
    vice = df[df["player"] != captain].sort_values("vc_score", ascending=False).iloc[0]["player"]

    return captain, vice