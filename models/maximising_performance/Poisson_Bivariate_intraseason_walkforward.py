"""
Cumulative, sliding INTRA-season walk-forward for the bivariate Poisson model
-- same ablation as RF_intraseason_walkforward.py, applied to
Poisson_Covariates_Bivariate.py (the project's best model) instead of RF.

Point of this script: same question as the RF version -- does giving the
model access to the current season's elapsed matches (fresher, more
stationary data) on top of the WINDOW prior full seasons measurably help,
compared to the season-level walk-forward that never touches the test season
until predicting the whole thing at once? If the RF result (a real
improvement, closing ~28% of the gap to Bet365) generalizes here too, that's
further evidence non-stationarity is a genuine factor across model types, not
an RF-specific quirk.

One extra reason this is worth checking for THIS model specifically: the
Poisson model was already close to its estimation "sweet spot" (12 params,
~100 observations/param on the season-level WINDOW=3 setup) -- unlike RF,
which had spare capacity to badly overfit, the Poisson model has much less
room to gain from fresher data if the bottleneck was never overfitting in
the first place. So a null result here (unlike RF) would be informative too:
it would suggest the earlier RF gain was about fixing tree-specific
overfitting to stale patterns, not about a universal "fresher data always
helps" effect.

Design: identical chunking scheme to RF_intraseason_walkforward.py -- each
test season split chronologically into N_CHUNKS equal pieces; chunk i trains
on the WINDOW prior full seasons plus every earlier chunk of the current
season (0..i-1), tests on chunk i. Reuses PoissonRegressionGoalsBivariate and
HOME_COVARIATES/AWAY_COVARIATES directly from Poisson_Covariates_Bivariate.py
so the model itself is identical, only the train/test split logic differs.

RESULT: essentially flat, unlike RF. Pooled log loss went from 0.9719
(season-level, same 3236 test matches) to 0.9711 here -- a 0.0008 change,
noise-level, nowhere near RF's 0.0085 improvement (0.9914 -> 0.9829).
Accuracy actually ticked down very slightly (54.05% -> 53.99%); win-rate
ticked up slightly (47.22% -> 47.78%). No metric moved meaningfully.

This is the informative null result flagged as possible above: RF had spare
capacity it was using to overfit to stale season-to-season patterns, so
fresher intra-season data gave it something real to fix. This model was
already well-specified for its parameter count (12 params, ~100
observations/param even under the season-level WINDOW=3 setup) -- it wasn't
overfitting to begin with, so there's much less slack for fresher data to
recover. Together with the RF result, this is a fairly clean piece of
evidence that non-stationarity specifically hurts high-capacity models (RF,
and probably XGBoost/SVM by extension) much more than it hurts a compact,
correctly-specified generative model -- it's not a universal "fresher data
always helps" effect, it's specifically a capacity-dependent one.
"""

import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # models/, for Poisson_Covariates_Bivariate

from Poisson_Covariates_Bivariate import PoissonRegressionGoalsBivariate, HOME_COVARIATES, AWAY_COVARIATES

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3     # prior full seasons used as the historical baseline, same as Poisson_Covariates_Bivariate.py
N_CHUNKS = 5   # how many chronological chunks each test season is split into


def report_comparison(metric_name, model_arr, baseline_arr, baseline_name):
    """Paired t-test + bootstrap CI + win-rate for model vs. one baseline, on one metric."""
    t_stat, p_value = stats.ttest_rel(model_arr, baseline_arr)
    print(f"\nPaired t-test ({metric_name}, model vs {baseline_name}): t={t_stat:.3f}, p={p_value:.6f}")

    diff = model_arr - baseline_arr
    rng = np.random.default_rng(42)
    n = len(diff)
    boot_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI for mean {metric_name} diff (model - {baseline_name}): [{ci_low:.4f}, {ci_high:.4f}]")

    model_wins = (model_arr < baseline_arr).sum()
    baseline_wins = (baseline_arr < model_arr).sum()
    print(
        f"Model better on {model_wins}/{n} matches ({model_wins/n:.2%}), "
        f"{baseline_name} better on {baseline_wins}/{n} matches ({baseline_wins/n:.2%})"
    )


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    seasons = sorted(all_df["Season"].unique())

    chunk_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct, all_b365_correct, all_avg_correct = [], [], []
    per_chunk_idx_ll = {i: [] for i in range(N_CHUNKS)}  # for the within-season trend

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

            proba = model.predict_proba(test_chunk)  # columns: [H, D, A]
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
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
            avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
            model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
            b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
            avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)
            model_pred = np.array(classes_order)[proba.argmax(axis=1)]
            b365_pred = np.array(classes_order)[fair_b365.argmax(axis=1)]
            avg_pred = np.array(classes_order)[fair_avg.argmax(axis=1)]

            all_model_ll.append(model_ll)
            all_b365_ll.append(b365_ll)
            all_avg_ll.append(avg_ll)
            all_model_brier.append(model_brier)
            all_b365_brier.append(b365_brier)
            all_avg_brier.append(avg_brier)
            all_model_correct.append(model_pred == y_true)
            all_b365_correct.append(b365_pred == y_true)
            all_avg_correct.append(avg_pred == y_true)
            per_chunk_idx_ll[c_idx].append(model_ll.mean())

            chunk_rows.append({
                "test_season": test_season,
                "chunk": c_idx,
                "n_train": len(train_df),
                "n_test": len(test_chunk),
                "model_ll": model_ll.mean(),
                "bet365_ll": b365_ll.mean(),
                "avg_ll": avg_ll.mean(),
            })

        print(f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)})")

    chunk_df = pd.DataFrame(chunk_rows)

    print(f"\n=== Within-season trend: mean model log loss by chunk index (0=start of season, {N_CHUNKS - 1}=end) ===")
    for c_idx in range(N_CHUNKS):
        vals = per_chunk_idx_ll[c_idx]
        if vals:
            print(f"chunk {c_idx}: mean_log_loss={np.mean(vals):.4f}  (n_seasons={len(vals)})")

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    avg_ll_all = np.concatenate(all_avg_ll)
    model_brier_all = np.concatenate(all_model_brier)
    b365_brier_all = np.concatenate(all_b365_brier)
    avg_brier_all = np.concatenate(all_avg_brier)
    model_correct_all = np.concatenate(all_model_correct)
    b365_correct_all = np.concatenate(all_b365_correct)
    avg_correct_all = np.concatenate(all_avg_correct)

    print(f"\n=== Pooled across all {len(chunk_df)} chunks ({len(model_ll_all)} test matches) ===")
    print(f"Model log loss:      {model_ll_all.mean():.4f}")
    print(f"Bet365 log loss:     {b365_ll_all.mean():.4f}")
    print(f"Avg-bookie log loss: {avg_ll_all.mean():.4f}")
    print(f"Model Brier:         {model_brier_all.mean():.4f}")
    print(f"Bet365 Brier:        {b365_brier_all.mean():.4f}")
    print(f"Avg-bookie Brier:    {avg_brier_all.mean():.4f}")
    print(f"Model accuracy (1X2):      {model_correct_all.mean():.2%}")
    print(f"Bet365 accuracy (1X2):     {b365_correct_all.mean():.2%}")
    print(f"Avg-bookie accuracy (1X2): {avg_correct_all.mean():.2%}")
    print(
        "\n(Compare against Poisson_Covariates_Bivariate.py's season-level result: "
        "0.9719 pooled log loss / 54.05% accuracy / 47.22% win-rate.)"
    )

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


if __name__ == "__main__":
    main()
