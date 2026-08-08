"""
Greedy forward selection of extra covariates for PoissonRegressionGoalsBivariate
(Poisson_Covariates_Bivariate.py, the project's best model), informed by
RF_feature_importance_engineered.py's Gini + permutation importance results.

Starting point is the CURRENT best covariate set (fixed, never removed):
    HOME_COVARIATES = ["Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_PPG"]
    AWAY_COVARIATES = ["Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_PPG"]

Candidate pool: base stat names that showed up as reliably important (Gini
and/or permutation top-15) but aren't already in the model, added as
mirrored pairs following the same convention as Home_Elo/Away_Elo -- own
value + opponent's same-stat value -- e.g. adding "ShotOnTargetDifference_Rolling5"
appends Home_ShotOnTargetDifference_Rolling5 and Away_ShotOnTargetDifference_Rolling5
to HOME_COVARIATES (and the mirrored order to AWAY_COVARIATES). "PPG" is
included as a candidate too, since the current model only ever gives each
equation its OWN team's PPG, never the opponent's -- unlike Elo/xGA, which
both already carry opponent information.

Search procedure, each round:
  1. For every remaining candidate, add it (mirrored pair) to the current
     covariate set and score with the FULL intra-season cumulative walk-forward
     (WINDOW=3, N_CHUNKS=5, all 12 seasons, 3236 pooled test matches --
     identical protocol to Poisson_Bivariate_intraseason_walkforward.py).
  2. Keep whichever single candidate produced the lowest pooled log loss.
  3. If that improvement is smaller than MIN_IMPROVEMENT, stop -- the search
     has converged (mirrors the "run to convergence" stopping rule used for
     the EWMA span sweep elsewhere in this project).
  4. Otherwise commit the candidate permanently and repeat with the shrunk pool.

Global module attributes on Poisson_Covariates_Bivariate are monkey-patched
per trial (HOME_COVARIATES/AWAY_COVARIATES are read as globals inside
fit()/predict_proba(), not passed as arguments) -- restored to the committed
covariate set at the end of each round.

RESULT: filled in after running -- see bottom of this docstring.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # models/, for Poisson_Covariates_Bivariate

import Poisson_Covariates_Bivariate as pcb

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3
N_CHUNKS = 5
MIN_IMPROVEMENT = 0.0005  # stop when the best candidate's log loss gain is smaller than this
MAX_ROUNDS = 8

BASE_HOME = ["Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_PPG"]
BASE_AWAY = ["Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_PPG"]

CANDIDATE_POOL = [
    "PPG",
    "ShotOnTargetDifference_Rolling5",
    "ShotOnTargetDifference_RollingTeam7",
    "Shots_Rolling5",
    "ShotDifference_RollingTeam7",
    "ShotsOnTarget_Rolling5",
    "ShotDifference_Rolling5",
    "deep_Rolling5",
    "GoalDifference_RollingTeam7",
    "GoalDifference_Rolling5",
    "xGA_RollingTeam7",
    "ShotAccuracy_Rolling5",
    "xG_RollingTeam7",
]


def pooled_log_loss(all_df, seasons, home_cov, away_cov):
    pcb.HOME_COVARIATES = home_cov
    pcb.AWAY_COVARIATES = away_cov

    all_ll = []
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

            model = pcb.PoissonRegressionGoalsBivariate().fit(train_df)
            proba = model.predict_proba(test_chunk)
            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=["H", "D", "A"])

            eps = 1e-15
            ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            all_ll.append(ll)

    return np.concatenate(all_ll).mean()


def add_candidate(home_cov, away_cov, name):
    return (
        home_cov + [f"Home_{name}", f"Away_{name}"],
        away_cov + [f"Away_{name}", f"Home_{name}"],
    )


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    seasons = sorted(all_df["Season"].unique())

    current_home, current_away = list(BASE_HOME), list(BASE_AWAY)
    current_ll = pooled_log_loss(all_df, seasons, current_home, current_away)
    base_ll = current_ll
    print(f"Base covariates ({len(current_home)} each side): log loss = {current_ll:.4f}")

    remaining = list(CANDIDATE_POOL)
    committed = []

    for round_num in range(1, MAX_ROUNDS + 1):
        if not remaining:
            print("\nNo candidates left -- stopping.")
            break

        print(f"\n--- Round {round_num}: evaluating {len(remaining)} candidates ---")
        round_results = []
        for name in remaining:
            trial_home, trial_away = add_candidate(current_home, current_away, name)
            ll = pooled_log_loss(all_df, seasons, trial_home, trial_away)
            delta = current_ll - ll
            round_results.append((name, ll, delta))
            print(f"  + {name:40s} log loss = {ll:.4f}  (delta = {delta:+.4f})")

        round_results.sort(key=lambda r: r[1])
        best_name, best_ll, best_delta = round_results[0]

        if best_delta < MIN_IMPROVEMENT:
            print(f"\nBest candidate this round ({best_name}, delta={best_delta:+.4f}) "
                  f"is below MIN_IMPROVEMENT={MIN_IMPROVEMENT} -- stopping.")
            break

        current_home, current_away = add_candidate(current_home, current_away, best_name)
        current_ll = best_ll
        committed.append(best_name)
        remaining.remove(best_name)
        print(f"\n==> Committed: {best_name}  (new pooled log loss = {current_ll:.4f})")

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Committed additions (in order): {committed}")
    print(f"Final HOME_COVARIATES: {current_home}")
    print(f"Final AWAY_COVARIATES: {current_away}")
    print(f"Final pooled log loss: {current_ll:.4f}  (base was {base_ll:.4f})")


if __name__ == "__main__":
    main()
