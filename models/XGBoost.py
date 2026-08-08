"""
XGBoost, 1X2 prediction, full feature set, with early stopping (see prior
history: the walk-forward-vs-market version had originally dropped early
stopping, which was the actual cause of its severe overfitting -- fixed by
watching a held-out validation fold and halting once validation stops
improving, same as every XGBoost run since).

Now on the "classic" intra-season cumulative sliding walk-forward (same
ablation as RF_intraseason_walkforward.py / Poisson_Bivariate_intraseason_
walkforward.py / FFNN.py / LogisticRegression.py / RF.py): each test season
is split into N_CHUNKS chronological pieces, and training grows every chunk
(prior WINDOW-VAL_SEASONS seasons + every earlier chunk of the current
season). The most recent of the WINDOW prior seasons is held out FIXED as
the early-stopping validation set (same VAL_SEASONS carve-out as before) --
it does not get folded into training as the season progresses, only the
current test season's elapsed chunks do.

RESULT: filled in after running -- see bottom of this docstring.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3        # total prior seasons pulled in (train + validation)
VAL_SEASONS = 1   # most recent of those WINDOW seasons, held out fixed for early stopping
N_CHUNKS = 5      # chronological chunks the current test season is split into
EARLY_STOPPING_ROUNDS = 20

drop_cols = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
]


def eval_probs(model, X, y_encoded, n_classes):
    """Log loss + Brier score for a fitted model against a labeled set (y already integer-encoded)."""
    proba = model.predict_proba(X)
    y_onehot = label_binarize(y_encoded, classes=range(n_classes))
    eps = 1e-15
    ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1)).mean()
    brier = ((proba - y_onehot) ** 2).sum(axis=1).mean()
    return ll, brier


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

    le = LabelEncoder()
    le.fit(all_df["FTR"])  # fit once on the full label set so encoding is consistent across every fold

    chunk_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct = []
    per_chunk_idx_ll = {i: [] for i in range(N_CHUNKS)}

    for i in range(WINDOW, len(seasons)):
        window_seasons = seasons[i - WINDOW:i]
        train_seasons = window_seasons[:-VAL_SEASONS]
        val_season_list = window_seasons[-VAL_SEASONS:]
        test_season = seasons[i]

        prior_df = all_df[all_df["Season"].isin(train_seasons)]
        val_df = all_df[all_df["Season"].isin(val_season_list)]
        season_df = all_df[all_df["Season"] == test_season].sort_values("Date").reset_index(drop=True)
        chunks = np.array_split(season_df, N_CHUNKS)

        X_val = val_df.drop(columns=drop_cols).fillna(0)
        y_val = le.transform(val_df["FTR"])

        for c_idx, test_chunk in enumerate(chunks):
            if test_chunk.empty:
                continue
            elapsed_chunks = pd.concat(chunks[:c_idx]) if c_idx > 0 else season_df.iloc[0:0]
            train_df = pd.concat([prior_df, elapsed_chunks])

            X_train = train_df.drop(columns=drop_cols).fillna(0)
            y_train = le.transform(train_df["FTR"])
            X_test = test_chunk.drop(columns=drop_cols).fillna(0)

            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            classes_order = le.classes_  # ['A', 'D', 'H'], alphabetical -- matches le.transform's integer order

            train_ll, _ = eval_probs(xgb, X_train, y_train, len(classes_order))
            val_ll, _ = eval_probs(xgb, X_val, y_val, len(classes_order))

            proba = xgb.predict_proba(X_test)

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

            y_true_encoded = le.transform(test_chunk["FTR"].values)
            y_onehot = label_binarize(y_true_encoded, classes=range(len(classes_order)))

            eps = 1e-15
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
            avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
            model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
            b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
            avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)
            model_pred = classes_order[proba.argmax(axis=1)]
            model_correct = (model_pred == test_chunk["FTR"].values)

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
                "best_iteration": xgb.best_iteration,
                "train_ll": train_ll,
                "val_ll": val_ll,
                "model_ll": model_ll.mean(),
                "bet365_ll": b365_ll.mean(),
                "avg_ll": avg_ll.mean(),
            })

        print(
            f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)}, "
            f"val season={val_season_list[0]})"
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
    print(f"\nMean best_iteration across chunks: {chunk_df['best_iteration'].mean():.1f} (max 200)")

    print(f"\nMean train log loss:      {chunk_df['train_ll'].mean():.4f}")
    print(f"Mean validation log loss: {chunk_df['val_ll'].mean():.4f}")
    print(
        f"Gap (val - train): {chunk_df['val_ll'].mean() - chunk_df['train_ll'].mean():.4f}  "
        "(large gap = overfitting)"
    )

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


if __name__ == "__main__":
    main()
