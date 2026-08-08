"""
Sweeps the EWMA span used for the xG/xGA rolling covariates
(Home_xG_Rolling5, Away_xG_Rolling5, Home_xGA_Rolling5, Away_xGA_Rolling5 --
the only rolling-window-dependent inputs Poisson_Covariates_Bivariate.py
actually uses; Home_TablePosDiff is not rolled, Elo is not rolled) from 3 to
10, to get empirical justification for span=5 instead of it being an
unverified inherited choice.

Only rebuilds the two affected raw stats (xG, xGA), venue-grouped
(Team+Venue), EWMA-weighted with the swept span -- everything else in the
dataset (Elo, TablePosDiff, Bet365/Avg odds, results) is left untouched.
Column names stay Home_xG_Rolling5/Away_xGA_Rolling5/etc. regardless of the
span actually used to compute them at each sweep step, purely so
PoissonRegressionGoalsBivariate (imported unmodified from
Poisson_Covariates_Bivariate.py) can consume them without any changes --
the "5" in those names is nominal during this sweep, not the true span.

For each span, runs the exact same WINDOW=3 walk-forward-vs-market
evaluation (Bet365 + avg-bookie, log loss + Brier) as
Poisson_Covariates_Bivariate.py, so results are directly comparable to the
project's reported 0.9719 log loss / 0.9711 Brier baseline (which used
span=5).

RESULT: filled in after running -- see bottom of this docstring.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, for rebuild_rolling_as_ewma / process_season_data

from Poisson_Covariates_Bivariate import PoissonRegressionGoalsBivariate, HOME_COVARIATES, AWAY_COVARIATES
from rebuild_rolling_as_ewma import build_team_centric

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 3
SPANS = list(range(3, 11))  # 3, 4, 5, ..., 10


def rebuild_xg_columns(matches, team_all, span):
    """Recompute the venue-grouped xG/xGA EWMA at the given span and merge
    them back onto a fresh copy of `matches`, overwriting the existing
    Home/Away_x{G,GA}_Rolling5 columns (name kept fixed -- see docstring)."""
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


def walk_forward_pooled(all_df):
    """Same evaluation as Poisson_Covariates_Bivariate.py's walk_forward_vs_market,
    trimmed to just what's needed for the sweep summary (pooled log loss/Brier,
    no per-fold printing -- there are 8 spans x 9 folds to get through)."""
    seasons = sorted(all_df["Season"].unique())
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]

        train_df = all_df[all_df["Season"].isin(train_seasons)]
        test_df = all_df[all_df["Season"] == test_season]

        model = PoissonRegressionGoalsBivariate().fit(train_df)
        proba = model.predict_proba(test_df)
        classes_order = ["H", "D", "A"]

        b365_overround = 1 / test_df["B365HomeOdds"] + 1 / test_df["B365DrawOdds"] + 1 / test_df["B365AwayOdds"]
        fair_b365 = np.column_stack([
            ((1 / test_df["B365HomeOdds"]) / b365_overround).values,
            ((1 / test_df["B365DrawOdds"]) / b365_overround).values,
            ((1 / test_df["B365AwayOdds"]) / b365_overround).values,
        ])

        avg_overround = 1 / test_df["AvgHomeOdds"] + 1 / test_df["AvgDrawOdds"] + 1 / test_df["AvgAwayOdds"]
        fair_avg = np.column_stack([
            ((1 / test_df["AvgHomeOdds"]) / avg_overround).values,
            ((1 / test_df["AvgDrawOdds"]) / avg_overround).values,
            ((1 / test_df["AvgAwayOdds"]) / avg_overround).values,
        ])

        y_true = test_df["FTR"].values
        y_onehot = label_binarize(y_true, classes=classes_order)

        eps = 1e-15
        model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
        b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
        avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
        model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
        b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
        avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)

        all_model_ll.append(model_ll)
        all_b365_ll.append(b365_ll)
        all_avg_ll.append(avg_ll)
        all_model_brier.append(model_brier)
        all_b365_brier.append(b365_brier)
        all_avg_brier.append(avg_brier)

    return {
        "model_ll": np.concatenate(all_model_ll),
        "b365_ll": np.concatenate(all_b365_ll),
        "avg_ll": np.concatenate(all_avg_ll),
        "model_brier": np.concatenate(all_model_brier),
        "b365_brier": np.concatenate(all_b365_brier),
        "avg_brier": np.concatenate(all_avg_brier),
    }


def main():
    # rebuild_rolling_as_ewma.build_team_centric() reads raw season CSVs
    # (e.g. "pl14-15.csv") and "scratch/leaguedata_<year>.json" via relative
    # paths, which only resolve correctly from the project root -- not from
    # models/, where this script actually lives.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    matches = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    seasons = sorted(matches["Season"].unique())

    print("Building team-centric raw stats once (shared across every span)...")
    team_all = build_team_centric(matches, seasons)

    results = []
    for span in SPANS:
        print(f"\n{'='*70}\nSPAN = {span}\n{'='*70}")
        df_span = rebuild_xg_columns(matches, team_all, span)
        pooled = walk_forward_pooled(df_span)

        n = len(pooled["model_ll"])
        model_ll_mean = pooled["model_ll"].mean()
        b365_ll_mean = pooled["b365_ll"].mean()
        avg_ll_mean = pooled["avg_ll"].mean()
        model_brier_mean = pooled["model_brier"].mean()

        t_stat, p_value = stats.ttest_rel(pooled["model_ll"], pooled["b365_ll"])
        model_wins = (pooled["model_ll"] < pooled["b365_ll"]).sum()

        print(
            f"span={span:2d}  n={n}  model_ll={model_ll_mean:.4f}  bet365_ll={b365_ll_mean:.4f}  "
            f"avg_ll={avg_ll_mean:.4f}  model_brier={model_brier_mean:.4f}  "
            f"win_rate={model_wins/n:.2%}  t={t_stat:.3f} p={p_value:.4f}"
        )

        results.append({
            "span": span,
            "n": n,
            "model_log_loss": model_ll_mean,
            "bet365_log_loss": b365_ll_mean,
            "avg_log_loss": avg_ll_mean,
            "model_brier": model_brier_mean,
            "win_rate_vs_bet365": model_wins / n,
        })

    results_df = pd.DataFrame(results)
    print(f"\n\n{'='*70}\nSWEEP SUMMARY (span=5 is the currently-deployed value)\n{'='*70}")
    print(results_df.round(4).to_string(index=False))

    best_row = results_df.loc[results_df["model_log_loss"].idxmin()]
    print(f"\nBest span by pooled log loss: {int(best_row['span'])} (log loss={best_row['model_log_loss']:.4f})")


if __name__ == "__main__":
    main()
