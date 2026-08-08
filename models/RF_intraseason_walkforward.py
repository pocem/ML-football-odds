"""
Cumulative, sliding INTRA-season walk-forward for Random Forest -- an ablation
against the season-level walk-forward used everywhere else in this project
(regression_models.ipynb's last RF cell: train on WINDOW prior full seasons,
test on the whole next season in one shot).

Point of this script: isolate which of the two suspected causes of RF's
underperformance is the bigger one -- (a) small sample size relative to tree
capacity, or (b) non-stationarity from season-to-season squad/manager/tactics
churn. Within a season, rosters are far more stable than across a season
boundary (only the January transfer window disrupts things), so if RF does
meaningfully better here than in the season-level walk-forward, that's
evidence non-stationarity was the bigger problem. If it doesn't, that points
back to sample size/model capacity as the dominant issue, since this design
doesn't give RF more data -- it gives it FRESHER data.

Design: each test season is split chronologically (by Date) into 10 equal
chunks (deciles). For chunk i (0-indexed), the model trains on the WINDOW
prior full seasons PLUS every earlier chunk of the current season (chunks
0..i-1, i.e. everything that happened before chunk i kicked off), and tests
on chunk i itself. This is "cumulative": the training set grows every step
as the season progresses, using the freshest same-season form on top of the
historical baseline, unlike the old approach which never touches any of the
test season until the whole thing is being predicted at once. Same feature
set, same RF hyperparameters, and same Bet365 comparison as the last RF cell
in regression_models.ipynb, so results are directly comparable.

RESULT: a real, meaningful improvement -- pooled log loss dropped from 0.9914
(season-level RF, same 3236 test matches across seasons 17-18..25-26) to
0.9829 here, closing about 28% of the gap to Bet365 (0.9615). Accuracy also
improved slightly (53.21% vs the season-level baseline). This supports
non-stationarity being a real contributor to RF's underperformance, not just
sample size -- fresher same-season data measurably helps, even though total
training-set size at any given step is similar to or smaller than before.

Caveat: the 0.9914 vs 0.9829 comparison is NOT a formal paired significance
test -- both evaluate the identical 3236 matches, but the per-match
predictions from the season-level RF run weren't retained in this session to
run a paired t-test between the two approaches directly (only each vs.
Bet365 was tested rigorously here). Treat the improvement as a strong signal,
not a proven-significant one, until that direct comparison is run.

The within-season trend (log loss by chunk index, chunk 0 = first ~38 matches
of the season, chunk 9 = last ~38) is noisy rather than a clean monotonic
"gets better as the season goes on" curve -- the pooled improvement comes
from the cumulative extra data overall, not from a strong specific-chunk
effect. Chunk 0's log loss (1.0145) looks worse than the season-level
baseline, but that's because it's measured on ONLY the first ~38 matches of
each season (early-season matches -- summer signings, new-manager bounce --
may just be harder to predict in general), not because this method
underperforms at that point; training data at chunk 0 is identical to the
season-level approach.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 3         # prior full seasons used as the historical baseline, same as regression_models.ipynb
N_CHUNKS = 5       # how many chronological chunks each test season is split into

drop_cols = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
]


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )


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

            X_train = train_df.drop(columns=drop_cols).fillna(0)
            y_train = train_df["FTR"]
            X_test = test_chunk.drop(columns=drop_cols).fillna(0)

            rf = make_rf()
            rf.fit(X_train, y_train)
            classes_order = rf.classes_

            proba = rf.predict_proba(X_test)

            b365_overround = 1 / test_chunk["B365HomeOdds"] + 1 / test_chunk["B365DrawOdds"] + 1 / test_chunk["B365AwayOdds"]
            fair_b365_dict = {
                "H": (1 / test_chunk["B365HomeOdds"]) / b365_overround,
                "D": (1 / test_chunk["B365DrawOdds"]) / b365_overround,
                "A": (1 / test_chunk["B365AwayOdds"]) / b365_overround,
            }
            fair_b365 = np.column_stack([fair_b365_dict[c].values for c in classes_order])

            avg_overround = 1 / test_chunk["AvgHomeOdds"] + 1 / test_chunk["AvgDrawOdds"] + 1 / test_chunk["AvgAwayOdds"]
            fair_avg_dict = {
                "H": (1 / test_chunk["AvgHomeOdds"]) / avg_overround,
                "D": (1 / test_chunk["AvgDrawOdds"]) / avg_overround,
                "A": (1 / test_chunk["AvgAwayOdds"]) / avg_overround,
            }
            fair_avg = np.column_stack([fair_avg_dict[c].values for c in classes_order])

            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=classes_order)
            eps = 1e-15
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
            avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
            model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
            b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
            avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)
            model_pred = classes_order[proba.argmax(axis=1)]
            b365_pred = classes_order[fair_b365.argmax(axis=1)]
            avg_pred = classes_order[fair_avg.argmax(axis=1)]

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
    print("\n(Compare against regression_models.ipynb's season-level RF: 0.9914 pooled log loss / market-implied win-rate 40.91%.)")

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


if __name__ == "__main__":
    main()
