"""
Builds dataset/all_seasons_14window_ppg.csv -- same as all_seasons_14window.csv,
but with the base Home_TablePosDiff/Away_TablePosDiff columns (ordinal league
rank) replaced by Home_PPG/Away_PPG/PPG_Difference (points-per-game, a real
interval measure -- the gap between 1st and 2nd isn't the same as the gap
between 19th and 20th, PPG doesn't have that distortion).

Only the BASE (non-rolling) position columns are swapped -- the rolling
TablePosDiff_Rolling5/RollingTeam7 features (used by the full-91-feature
RF/XGBoost/SVM scripts, not by the Poisson covariates model) are left as-is.
Ask for those to be swapped too if you want full parity.
"""

import pandas as pd

from process_season_data import load_data, add_points_per_game

INPUT_FILE = "dataset/all_seasons_14window.csv"
OUTPUT_FILE = "dataset/all_seasons_14window_ppg.csv"


def main():
    matches = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    seasons = sorted(matches["Season"].unique())

    print("Dropping base Home_TablePosDiff/Away_TablePosDiff columns")
    matches = matches.drop(columns=["Home_TablePosDiff", "Away_TablePosDiff"])

    ppg_frames = []
    prior_ppg = None
    for season in seasons:
        raw = load_data(f"pl{season}.csv")
        raw, prior_ppg = add_points_per_game(raw, prior_ppg=prior_ppg)
        ppg_frames.append(raw[["Date", "Time", "HomeTeam", "AwayTeam", "HomePPG", "AwayPPG", "PPGDiff"]])

    ppg_all = pd.concat(ppg_frames, ignore_index=True)
    ppg_all = ppg_all.rename(columns={
        "HomePPG": "Home_PPG",
        "AwayPPG": "Away_PPG",
        "PPGDiff": "PPG_Difference",
    })

    matches = matches.merge(ppg_all, on=["Date", "Time", "HomeTeam", "AwayTeam"], how="left")

    missing = matches[["Home_PPG", "Away_PPG", "PPG_Difference"]].isna().any(axis=1).sum()
    print(f"Rows missing PPG after merge: {missing} / {len(matches)}")

    matches.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {matches.shape} to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
