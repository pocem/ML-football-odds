"""
Logistic Regression, 1X2 prediction, full feature set, using the
hyperparameters selected by tuning/LR_TUNING.PY's 1-SE-rule RandomizedSearchCV
(C=0.00101, penalty='l2') -- strong regularization, which makes sense given
a linear model exposed to all correlated raw features at once, with no
tree-style implicit feature selection to fall back on.

Now on the "classic" intra-season cumulative sliding walk-forward (same
ablation as RF_intraseason_walkforward.py / Poisson_Bivariate_intraseason_
walkforward.py / FFNN.py): each test season is split into N_CHUNKS
chronological pieces, and training grows every chunk (prior WINDOW seasons +
every earlier chunk of the current season), instead of the old season-level
version that only ever trained on whole prior seasons and tested the whole
next season in one shot. No separate validation season here (unlike FFNN,
LR doesn't need one for early stopping) -- matches RF/Poisson's simpler
intra-season pattern: WINDOW prior seasons is the whole training base before
any current-season chunks are folded in.

StandardScaler is required (unlike RF/XGBoost) since LR is a linear model on
raw features spanning wildly different scales (Elo in the hundreds vs. PPG
in low single digits) -- fit on TRAIN only per fold, like everywhere else.

RESULT (WINDOW=3, N_CHUNKS=5, pooled over 3236 matches): a real improvement
over the season-level version -- pooled log loss dropped from 0.9902 to
0.9752, closing a meaningful chunk of the gap to Bet365 (0.9615) and landing
close to the Poisson bivariate model's intra-season result (0.9695). Both
paired t-tests are still significant (p<0.00001, model still loses to the
market), but by a smaller margin than before. Notably better than RF's
intra-season result (0.9829-0.9835) and clearly better than XGBoost's/FFNN's
intra-season numbers -- LR remains the strongest of the "properly regularized,
not exotic" model family, now doing even better once given access to
same-season in-progress data like the other intra-season scripts.

Within-season trend is not fully monotonic: log loss dips notably mid-season
(chunk 2: 1.017) before improving sharply late (chunks 3-4: ~0.95), unlike a
clean "always gets better with more data" curve -- worth keeping in mind
before reading too much into any single chunk's number.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3     # prior full seasons used as the historical baseline
N_CHUNKS = 5   # how many chronological chunks each test season is split into

LR_PARAMS = {"C": 0.0010090061869151559, "penalty": "l2"}

drop_cols = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
]


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
    print(f"Feature count: {all_df.drop(columns=drop_cols).shape[1]}")
    print(f"LR params: {LR_PARAMS}")

    chunk_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct = []
    per_chunk_idx_ll = {i: [] for i in range(N_CHUNKS)}

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

            X_train_raw = train_df.drop(columns=drop_cols).fillna(0)
            y_train = train_df["FTR"]
            X_test_raw = test_chunk.drop(columns=drop_cols).fillna(0)

            scaler = StandardScaler().fit(X_train_raw)
            X_train = scaler.transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            lr = LogisticRegression(max_iter=2000, solver="saga", random_state=42, **LR_PARAMS)
            lr.fit(X_train, y_train)
            classes_order = lr.classes_

            proba = lr.predict_proba(X_test)

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
            model_correct = (model_pred == y_true)

            all_model_ll.append(model_ll)
            all_b365_ll.append(b365_ll)
            all_avg_ll.append(avg_ll)
            all_model_brier.append(model_brier)
            all_b365_brier.append(b365_brier)
            all_avg_brier.append(avg_brier)
            all_model_correct.append(model_correct)
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

        print(
            f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)})"
        )

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

    print(f"\n=== Pooled across all {len(chunk_df)} chunks ({len(model_ll_all)} test matches) ===")
    print(f"Model log loss:      {model_ll_all.mean():.4f}")
    print(f"Bet365 log loss:     {b365_ll_all.mean():.4f}")
    print(f"Avg-bookie log loss: {avg_ll_all.mean():.4f}")
    print(f"Model Brier:         {model_brier_all.mean():.4f}")
    print(f"Bet365 Brier:        {b365_brier_all.mean():.4f}")
    print(f"Avg-bookie Brier:    {avg_brier_all.mean():.4f}")
    print(f"Model accuracy (1X2): {model_correct_all.mean():.4f}")

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


if __name__ == "__main__":
    main()
