"""
Sweeps the EWMA span used for the xG/xGA rolling covariates
(Home_xG_Rolling5, Away_xG_Rolling5, Home_xGA_Rolling5, Away_xGA_Rolling5)
from 5 upward until convergence (+2 more points past that), evaluated with
the exact intra-season cumulative sliding walk-forward from
Poisson_Bivariate_intraseason_walkforward.py (WINDOW=3, N_CHUNKS=5) -- not
the season-level harness models/Poisson_Bivariate_rolling_window_sweep.py
already covered.

Only the two affected raw stats (xG, xGA), venue-grouped (Team+Venue),
EWMA-weighted at the swept span, are rebuilt -- everything else (Elo,
TablePosDiff, Bet365/Avg odds, results) is read from the dataset as-is.
Column names stay Home_xG_Rolling5/etc. regardless of the span actually
used at each step, so PoissonRegressionGoalsBivariate (imported unmodified)
can consume them without changes -- the current dataset on disk has these
columns baked in at span=10 (from the last full-dataset rebuild), but this
script overrides them per sweep step and ignores whatever is currently saved.

Convergence rule: track the running minimum pooled log loss; if PATIENCE
consecutive spans pass without a new minimum, the sweep is done -- run 2
more points past the current position, then stop.

Produces one plot: span vs. pooled model log loss (with Bet365/avg-bookie
as flat reference lines), saved to images/poisson_intraseason_span_sweep.png.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from Poisson_Covariates_Bivariate import PoissonRegressionGoalsBivariate
from rebuild_rolling_as_ewma import build_team_centric

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 3       # same as Poisson_Bivariate_intraseason_walkforward.py
N_CHUNKS = 5     # same as Poisson_Bivariate_intraseason_walkforward.py
START_SPAN = 5
PATIENCE = 3     # consecutive non-improving spans before declaring convergence
EXTRA_POINTS = 2 # additional spans to run past the convergence point
MAX_SPAN = 40    # safety cap
OUTPUT_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\images\poisson_intraseason_span_sweep.png"


def rebuild_xg_columns(matches, team_all, span):
    team_all = team_all.copy()
    for col in ["xG", "xGA"]:
        team_all[f"{col}_Rolling5"] = (
            team_all.groupby(["Team", "Venue"])[col]
            .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
        )

    roll_cols = ["xG_Rolling5", "xGA_Rolling5"]
    home_roll = (
        team_all[team_all["Venue"] == "H"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in roll_cols}})
    )
    away_roll = (
        team_all[team_all["Venue"] == "A"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in roll_cols}})
    )

    out = matches.drop(columns=[
        "Home_xG_Rolling5", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_xGA_Rolling5"
    ])
    out = out.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    out = out.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")
    return out


def intraseason_pooled_ll(all_df):
    """Exact intra-season cumulative sliding walk-forward from
    Poisson_Bivariate_intraseason_walkforward.py, trimmed to just what's
    needed here: pooled model/Bet365/avg-bookie log loss arrays."""
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    seasons = sorted(all_df["Season"].unique())

    all_model_ll, all_b365_ll, all_avg_ll = [], [], []

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]

        prior_df = all_df[all_df["Season"].isin(train_seasons)]
        season_df = all_df[all_df["Season"] == test_season].sort_values("Date").reset_index(drop=True)
        chunks = np.array_split(season_df, N_CHUNKS)

        for c_idx, test_chunk in enumerate(chunks):
            if test_chunk.empty:
                continue
            elapsed_chunks = pd.concat(chunks[:c_idx]) if c_idx > 0 else season_df.iloc[0:0]
            train_df = pd.concat([prior_df, elapsed_chunks])

            model = PoissonRegressionGoalsBivariate().fit(train_df)
            proba = model.predict_proba(test_chunk)
            classes_order = ["H", "D", "A"]

            b365_overround = 1 / test_chunk["B365HomeOdds"] + 1 / test_chunk["B365DrawOdds"] + 1 / test_chunk["B365AwayOdds"]
            fair_b365 = np.column_stack([
                ((1 / test_chunk["B365HomeOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365DrawOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365AwayOdds"]) / b365_overround).values,
            ])
            avg_overround = 1 / test_chunk["AvgHomeOdds"] + 1 / test_chunk["AvgDrawOdds"] + 1 / test_chunk["AvgAwayOdds"]
            fair_avg = np.column_stack([
                ((1 / test_chunk["AvgHomeOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgDrawOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgAwayOdds"]) / avg_overround).values,
            ])

            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=classes_order)
            eps = 1e-15
            all_model_ll.append(-np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1)))
            all_b365_ll.append(-np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1)))
            all_avg_ll.append(-np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1)))

    return (
        np.concatenate(all_model_ll).mean(),
        np.concatenate(all_b365_ll).mean(),
        np.concatenate(all_avg_ll).mean(),
    )


def main():
    # rebuild_rolling_as_ewma.build_team_centric() reads raw season CSVs and
    # scratch/leaguedata_<year>.json via relative paths, which only resolve
    # correctly from the project root -- not from models/, where this script lives.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    matches = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    seasons = sorted(matches["Season"].unique())

    print("Building team-centric raw stats once (shared across every span)...")
    team_all = build_team_centric(matches, seasons)

    results = []
    best_ll = np.inf
    best_span = None
    no_improve_count = 0
    span = START_SPAN
    extra_remaining = None  # becomes a countdown once convergence is detected

    while True:
        df_span = rebuild_xg_columns(matches, team_all, span)
        model_ll, b365_ll, avg_ll = intraseason_pooled_ll(df_span)
        print(f"span={span:2d}  model_ll={model_ll:.4f}  bet365_ll={b365_ll:.4f}  avg_ll={avg_ll:.4f}")
        results.append({"span": span, "model_log_loss": model_ll, "bet365_log_loss": b365_ll, "avg_log_loss": avg_ll})

        if model_ll < best_ll - 1e-5:
            best_ll = model_ll
            best_span = span
            no_improve_count = 0
        else:
            no_improve_count += 1

        if extra_remaining is not None:
            extra_remaining -= 1
            if extra_remaining <= 0:
                break
        elif no_improve_count >= PATIENCE:
            print(f"\nConverged: best span={best_span} (log loss={best_ll:.4f}), "
                  f"no improvement for {PATIENCE} consecutive spans. Running {EXTRA_POINTS} more.")
            extra_remaining = EXTRA_POINTS

        span += 1
        if span > MAX_SPAN:
            print(f"\nHit MAX_SPAN={MAX_SPAN} safety cap without clean convergence, stopping.")
            break

    results_df = pd.DataFrame(results)
    print(f"\n{'='*70}\nSWEEP SUMMARY\n{'='*70}")
    print(results_df.round(4).to_string(index=False))
    print(f"\nBest span by pooled log loss: {best_span} (log loss={best_ll:.4f})")

    plt.figure(figsize=(9, 6))
    plt.plot(results_df["span"], results_df["model_log_loss"], marker="o", label="Model log loss (intra-season)")
    plt.axhline(results_df["bet365_log_loss"].iloc[0], color="gray", linestyle="--", alpha=0.6, label="Bet365")
    plt.axhline(results_df["avg_log_loss"].iloc[0], color="lightgray", linestyle=":", alpha=0.8, label="Avg-bookie")
    plt.axvline(best_span, color="green", linestyle="--", alpha=0.5, label=f"Best span = {best_span}")
    plt.xlabel("EWMA span (xG/xGA rolling covariates)")
    plt.ylabel("Pooled log loss (intra-season walk-forward)")
    plt.title("Poisson bivariate (intra-season): log loss by rolling-window span")
    plt.xticks(results_df["span"])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
