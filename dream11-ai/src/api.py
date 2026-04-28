from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .preprocess import load_data, clean_data
from .feature_engineering import create_player_features, create_ball_features, merge_features
from .predict import load_models, ensemble_predict
from .gl_strategy import generate_gl_teams
from .captain import select_captain_vice

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def home():
    return {"status": "Dream11 AI running"}

@app.get("/generate-gl-teams")
def generate_gl(match_id: int = None):

    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)

    bat_feat = create_player_features(bat)
    ball_feat = create_ball_features(ball)

    df = merge_features(bat_feat, ball_feat)

    xgb, lstm = load_models()
    df = ensemble_predict(df, xgb, lstm)

    # add GL features
    df["ceiling"] = df["avg_last10"] + df["std_last10"]
    df["risk"] = df["std_last10"]
    df["ownership"] = df["final_pred"].rank(pct=True)
    df["differential_score"] = (1 - df["ownership"]) * df["ceiling"]

    teams = generate_gl_teams(df, 5)

    output = []

    for t in teams:
        c, vc = select_captain_vice(t)

        output.append({
            "players": t["player"].tolist(),
            "captain": c,
            "vice_captain": vc,
            "match_id": match_id
        })

    return output