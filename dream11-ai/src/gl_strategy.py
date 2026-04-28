import numpy as np
from .optimizer import optimize_team

def generate_gl_teams(df, n_teams=5):

    teams = []

    for i in range(n_teams):

        temp = df.copy()

        if i == 0:
            # SAFE TEAM
            temp["score"] = temp["final_pred"]

        elif i == 1:
            # BALANCED TEAM
            temp["score"] = (
                0.7 * temp["final_pred"] +
                0.3 * temp["consistency"]
            )

        else:
            # AGGRESSIVE (GL TEAM)
            temp["score"] = (
                0.5 * temp["final_pred"] +
                0.5 * temp["differential_score"]
            )

            # add randomness for diversity
            temp["score"] += np.random.normal(0, 5, len(temp))

        temp["final_pred"] = temp["score"]

        team = optimize_team(temp)

        teams.append(team)

    return teams