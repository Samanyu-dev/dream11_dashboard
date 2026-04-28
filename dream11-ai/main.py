"""
Dream11 AI — Full Pipeline Runner
Trains all models end-to-end and runs a sample prediction.
Usage: python main.py [--team1 CSK --team2 MI]
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline(team1: str = None, team2: str = None):
    from preprocess import load_data, clean_data, filter_teams
    from feature_engineering import create_player_features, create_ball_features, merge_features
    from train_model import train_model
    from dl_model import train_lstm
    from predict import load_models, ensemble_predict, aggregate_predictions
    from optimizer import optimize_team
    from captain import select_captain_vice
    from gl_strategy import generate_gl_teams, simulate_ownership
    from insights import generate_insights

    logger.info("=" * 60)
    logger.info("DREAM11 AI — FULL PIPELINE")
    logger.info("=" * 60)

    # 1. Load & clean
    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)

    # 2. Features (full dataset for training)
    bat_feat  = create_player_features(bat)
    ball_feat = create_ball_features(ball)
    df_full   = merge_features(bat_feat, ball_feat)

    # 3. Train models
    xgb, lgb_m, rf, scaler, feats = train_model(df_full)
    lstm = train_lstm(df_full)

    logger.info("✅ Training complete")

    # 4. If teams given, run prediction
    if team1 and team2:
        logger.info(f"\nGenerating team for {team1} vs {team2} …")
        bat_f, ball_f = filter_teams(bat, ball, team1, team2)
        bf   = create_player_features(bat_f)
        blf  = create_ball_features(ball_f)
        df_m = merge_features(bf, blf)

        models = load_models()
        df_m   = ensemble_predict(df_m, models)
        df_agg = simulate_ownership(aggregate_predictions(df_m))
        df_agg["differential_score"] = (1 - df_agg["ownership"] / 100) * df_agg["ceiling"]

        best = optimize_team(df_agg)
        cap, vc = select_captain_vice(best)

        print("\n" + "=" * 55)
        print(f"🔥 BEST DREAM11 TEAM: {team1} vs {team2}")
        print("=" * 55)
        for _, row in best.iterrows():
            tag = f"[{row['role_tag']}]" if row.get("role_tag") else ""
            print(f"  {tag:4s} {row['player']:<25} {row['team']:<6} pred={row['final_pred']:.1f}")
        print(f"\n  Captain: {cap}  |  Vice-Captain: {vc}")

        print("\n💡 INSIGHTS")
        for ins in generate_insights(df_agg)[:5]:
            print(" ", ins)

        print("\n🎲 GL TEAMS")
        gl = generate_gl_teams(df_agg, n_teams=3)
        for t in gl:
            print(f"\n  Team {t['team_no']} [{t['strategy']}]")
            print(f"  C: {t['captain']}  VC: {t['vice_captain']}")
            print(f"  Players: {', '.join(t['players'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dream11 AI Pipeline")
    parser.add_argument("--team1", default=None)
    parser.add_argument("--team2", default=None)
    args = parser.parse_args()
    run_pipeline(args.team1, args.team2)