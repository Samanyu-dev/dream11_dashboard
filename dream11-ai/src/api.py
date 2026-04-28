"""
Dream11 AI — FastAPI Backend
Endpoints: /predict-team  /generate-gl-teams  /player-stats  /match-insights
"""
import logging
import traceback
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import load_data, clean_data, filter_teams
from feature_engineering import create_player_features, create_ball_features, merge_features
from predict import load_models, ensemble_predict, aggregate_predictions
from optimizer import optimize_team
from gl_strategy import generate_gl_teams, simulate_ownership
from captain import select_captain_vice, captain_stats
from insights import generate_insights, player_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Dream11 AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Shared state: load data & models once ────────────────────────────────────
@lru_cache(maxsize=1)
def _get_data():
    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)
    return ball, bat, match


@lru_cache(maxsize=1)
def _get_models():
    return load_models()


def _build_df(team1: str, team2: str):
    ball, bat, match = _get_data()
    bat_f, ball_f = filter_teams(bat, ball, team1, team2)
    if bat_f.empty:
        raise HTTPException(404, f"No players found for {team1} vs {team2}")
    bat_feat  = create_player_features(bat_f)
    ball_feat = create_ball_features(ball_f)
    df = merge_features(bat_feat, ball_feat)
    models = _get_models()
    df = ensemble_predict(df, models)
    df = simulate_ownership(aggregate_predictions(df))
    df["differential_score"] = (1 - df["ownership"] / 100) * df["ceiling"]
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Dream11 AI v2.0 running", "endpoints": [
        "/predict-team", "/generate-gl-teams", "/player-stats", "/match-insights"
    ]}


@app.get("/predict-team")
def predict_team(
    team1: str = Query(..., example="CSK"),
    team2: str = Query(..., example="MI"),
    strategy: str = Query("balanced", enum=["safe", "balanced", "aggressive"]),
):
    try:
        df = _build_df(team1, team2)
        team_df = optimize_team(df)
        cap, vc = select_captain_vice(team_df, strategy=strategy)
        c_stats = captain_stats(team_df, cap, vc)
        return {
            "match": f"{team1} vs {team2}",
            "strategy": strategy,
            "captain_stats": c_stats,
            "players": team_df[[
                "player", "team", "role", "credits",
                "final_pred", "ceiling", "consistency", "role_tag"
            ]].to_dict("records"),
            "predicted_total": round(float(team_df["effective_pred"].sum()), 1) if "effective_pred" in team_df else 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.get("/generate-gl-teams")
def gen_gl(
    team1: str = Query(..., example="CSK"),
    team2: str = Query(..., example="MI"),
    n:     int = Query(5, ge=1, le=10),
):
    try:
        df = _build_df(team1, team2)
        teams = generate_gl_teams(df, n_teams=n)
        return {"match": f"{team1} vs {team2}", "teams": teams}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.get("/player-stats")
def player_stats(
    team1: str = Query(..., example="CSK"),
    team2: str = Query(..., example="MI"),
    player: Optional[str] = Query(None),
):
    try:
        df = _build_df(team1, team2)
        if player:
            return player_report(df, player)
        cols = ["player", "team", "role", "final_pred", "avg_last5", "avg_last10",
                "ceiling", "floor", "consistency", "std_last10", "ownership"]
        avail = [c for c in cols if c in df.columns]
        return {"players": df[avail].sort_values("final_pred", ascending=False).to_dict("records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/match-insights")
def match_insights(
    team1: str = Query(..., example="CSK"),
    team2: str = Query(..., example="MI"),
):
    try:
        df = _build_df(team1, team2)
        return {
            "match": f"{team1} vs {team2}",
            "insights": generate_insights(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))