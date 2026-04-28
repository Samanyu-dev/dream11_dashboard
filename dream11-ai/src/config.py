"""
Dream11 AI — Configuration
All paths, constants, and hyperparameters in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(BASE_DIR, "models")
DATA_PATH   = os.path.join(BASE_DIR)          # CSVs live next to this file
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")

# ── Data files ───────────────────────────────────────────────────────────────
BALL_CSV    = os.path.join(DATA_PATH, "all_matches_ball_by_ball1.csv")
BATTING_CSV = os.path.join(DATA_PATH, "Batting_data.csv")
IPL_CSV     = os.path.join(DATA_PATH, "ipl_data.csv")

# ── Feature Engineering ──────────────────────────────────────────────────────
ROLLING_WINDOWS   = [5, 10, 20]          # rolling window sizes
SEQ_LEN           = 15                   # LSTM sequence length

# ── Ensemble weights ─────────────────────────────────────────────────────────
W_XGB  = 0.50
W_LSTM = 0.30
W_RF   = 0.20

# ── Dream11 constraints ──────────────────────────────────────────────────────
TEAM_SIZE      = 11
BUDGET         = 100
MAX_FROM_TEAM  = 7
ROLE_LIMITS = {
    "WK":   (1, 4),
    "BAT":  (3, 6),
    "AR":   (1, 4),
    "BOWL": (3, 6),
}

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN_SEASONS = list(range(2008, 2023))  # train on 2008–2022
VAL_SEASONS   = [2022]
TEST_SEASONS  = [2023]
RANDOM_SEED   = 42

# ── GL Strategy ──────────────────────────────────────────────────────────────
N_GL_TEAMS = 5