"""
Dream11 AI — Quick Team Generator CLI
Usage: python generate_team.py CSK MI [--strategy balanced] [--gl 5]
"""
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate a Dream11 team")
    parser.add_argument("team1")
    parser.add_argument("team2")
    parser.add_argument("--strategy", default="balanced", choices=["safe", "balanced", "aggressive"])
    parser.add_argument("--gl", type=int, default=0, help="Also generate N Grand League teams")
    args = parser.parse_args()

    from preprocess import load_data, clean_data, filter_teams
    from feature_engineering import create_player_features, create_ball_features, merge_features
    from predict import load_models, ensemble_predict, aggregate_predictions
    from optimizer import optimize_team
    from captain import select_captain_vice
    from gl_strategy import generate_gl_teams, simulate_ownership
    from insights import generate_insights

    team1, team2 = args.team1.upper(), args.team2.upper()
    print(f"\n🏏  Dream11 AI — {team1} vs {team2}\n")

    ball, bat, match = load_data()
    ball, bat, match = clean_data(ball, bat, match)

    bat_f, ball_f = filter_teams(bat, ball, team1, team2)
    if bat_f.empty:
        print(f"❌ No players found for {team1} or {team2}. Check team names.")
        sys.exit(1)

    bat_feat  = create_player_features(bat_f)
    ball_feat = create_ball_features(ball_f)
    df = merge_features(bat_feat, ball_feat)

    models = load_models()
    df = ensemble_predict(df, models)
    df = simulate_ownership(aggregate_predictions(df))
    df["differential_score"] = (1 - df["ownership"] / 100) * df["ceiling"]

    # Best team
    team_df = optimize_team(df)
    cap, vc = select_captain_vice(team_df, strategy=args.strategy)

    print("=" * 55)
    print(f"🔥  BEST TEAM  [{args.strategy.upper()}]")
    print("=" * 55)
    for _, row in team_df.iterrows():
        tag = f"[{row['role_tag']}]" if row.get("role_tag") else "    "
        print(f"  {tag:4s} {row['player']:<28} {row.get('team','?'):<6} {row['final_pred']:.1f} pts")
    print(f"\n  Captain:      {cap}")
    print(f"  Vice-Captain: {vc}")
    print(f"\n  Est. team score: {team_df.get('effective_pred', team_df['final_pred']).sum():.1f} pts")

    print("\n💡  AI INSIGHTS")
    for ins in generate_insights(df)[:6]:
        print(f"  {ins}")

    if args.gl > 0:
        print(f"\n🎲  GRAND LEAGUE TEAMS ({args.gl})")
        gl_teams = generate_gl_teams(df, n_teams=args.gl)
        for t in gl_teams:
            print(f"\n  ── Team {t['team_no']} [{t['strategy']}] ──")
            print(f"  C: {t['captain']}  VC: {t['vice_captain']}")
            print(f"  Differentials: {', '.join(t['differentials'])}")
            print(f"  Players: {', '.join(t['players'])}")


if __name__ == "__main__":
    main()