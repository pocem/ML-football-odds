"""
SVM classifier for 1X2 prediction, same walk-forward-vs-Bet365 harness as the
RF/XGBoost notebooks (WINDOW=3, VAL_SEASONS=1, same drop_cols, same market
comparison) so results are directly comparable to those and to the Poisson
models. Unlike RF/XGBoost, SVC needs feature scaling (StandardScaler, fit on
TRAIN only per fold, like every other per-fold transform in this project) --
distances/margins are meaningless on raw features where Elo is in the
hundreds and TablePosDiff is +/-19.

Why try this at all, given RF/XGBoost already lost to the market by a wide
margin: SVM's regularization is fundamentally different from a tree
ensemble's. A tree ensemble's only defense against overfitting is limiting
depth/leaf size; a max-margin classifier is regularized by construction (it
optimizes for the widest separating margin, not by memorizing training
points), which is closer in spirit to the Poisson model's built-in structure
than to RF/XGBoost's brute-force flexibility. Worth an honest try given how
much this project has found "less flexible, more structured" beating "more
flexible" on this small, noisy dataset.

RESULTS (WINDOW=3, VAL_SEASONS=1, vs Bet365, pooled over 3236 matches):
  - First try, all 91 engineered features (StandardScaler-d, same as RF/
    XGBoost get): log loss 1.0121, accuracy 51.58%, val-train gap 0.173 --
    the "SVM is regularized by construction" hypothesis did NOT hold up in
    practice; it overfit even more than RF did (RF's gap was 0.128).
  - Restricted to the same 8 covariates as Poisson_Covariates_Bivariate.py
    (below): log loss improved to 1.0048, accuracy 52.69%, and the val-train
    gap shrank sharply to 0.069. RBF-kernel SVMs are known to suffer from
    curse-of-dimensionality (similarity/distance gets diluted by irrelevant
    dimensions, unlike trees which do implicit feature selection at each
    split) -- this is exactly that effect, confirmed empirically.
  - Still behind RF (0.9914) and well behind Bet365 (0.9615) and the Poisson
    bivariate model (0.9719, the project's best). Kept as the restricted-
    feature version below since it's the strictly better of the two, but
    this isn't a competitive model for this dataset either way.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 4
VAL_SEASONS = 1

FEATURES = [
    "Home_Elo", "Away_Elo",
    "Home_xG_Rolling5", "Away_xG_Rolling5",
    "Home_xGA_Rolling5", "Away_xGA_Rolling5",
    "Home_TablePosDiff", "Away_TablePosDiff",
]


class WalkForwardValidator:

    def __init__(self, seasons):
        self.seasons = seasons

    def split(self, meta, window, val_seasons=1):
        unique_seasons = sorted(meta["Season"].unique())
        for i in range(window, len(unique_seasons)):
            window_seasons = unique_seasons[i - window:i]
            train_seasons = window_seasons[:-val_seasons]
            val_season_list = window_seasons[-val_seasons:]
            test_season = unique_seasons[i]

            train_idx = meta[meta["Season"].isin(train_seasons)].index
            val_idx = meta[meta["Season"].isin(val_season_list)].index
            test_idx = meta[meta["Season"] == test_season].index

            yield (train_idx, val_idx, test_idx, train_seasons, val_season_list, test_season)


def eval_probs(model, X, y, classes_order):
    """Log loss + Brier score for a fitted model against a labeled set."""
    proba = model.predict_proba(X)
    y_onehot = label_binarize(y, classes=classes_order)
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


def walk_forward_vs_market(all_df):
    validator = WalkForwardValidator(seasons=sorted(all_df["Season"].unique()))

    fold_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct, all_b365_correct, all_avg_correct = [], [], []

    for train_idx, val_idx, test_idx, train_seasons, val_season_list, test_season in validator.split(
        all_df, window=WINDOW, val_seasons=VAL_SEASONS
    ):
        train_df = all_df.loc[train_idx]
        val_df = all_df.loc[val_idx]
        test_df = all_df.loc[test_idx]

        X_train_raw = train_df[FEATURES].fillna(0)
        y_train = train_df["FTR"]
        X_val_raw = val_df[FEATURES].fillna(0)
        y_val = val_df["FTR"]
        X_test_raw = test_df[FEATURES].fillna(0)

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)

        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        svm.fit(X_train, y_train)
        classes_order = svm.classes_

        train_ll, _ = eval_probs(svm, X_train, y_train, classes_order)
        val_ll, _ = eval_probs(svm, X_val, y_val, classes_order)

        proba = svm.predict_proba(X_test)

        b365_overround = 1 / test_df["B365HomeOdds"] + 1 / test_df["B365DrawOdds"] + 1 / test_df["B365AwayOdds"]
        fair_b365_dict = {
            "H": (1 / test_df["B365HomeOdds"]) / b365_overround,
            "D": (1 / test_df["B365DrawOdds"]) / b365_overround,
            "A": (1 / test_df["B365AwayOdds"]) / b365_overround,
        }
        fair_b365 = np.column_stack([fair_b365_dict[c].values for c in classes_order])

        avg_overround = 1 / test_df["AvgHomeOdds"] + 1 / test_df["AvgDrawOdds"] + 1 / test_df["AvgAwayOdds"]
        fair_avg_dict = {
            "H": (1 / test_df["AvgHomeOdds"]) / avg_overround,
            "D": (1 / test_df["AvgDrawOdds"]) / avg_overround,
            "A": (1 / test_df["AvgAwayOdds"]) / avg_overround,
        }
        fair_avg = np.column_stack([fair_avg_dict[c].values for c in classes_order])

        y_true = test_df["FTR"].values
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
        model_correct = (model_pred == y_true)
        b365_correct = (b365_pred == y_true)
        avg_correct = (avg_pred == y_true)

        all_model_ll.append(model_ll)
        all_b365_ll.append(b365_ll)
        all_avg_ll.append(avg_ll)
        all_model_brier.append(model_brier)
        all_b365_brier.append(b365_brier)
        all_avg_brier.append(avg_brier)
        all_model_correct.append(model_correct)
        all_b365_correct.append(b365_correct)
        all_avg_correct.append(avg_correct)

        fold_rows.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            "n_test": len(test_df),
            "train_log_loss": train_ll,
            "val_log_loss": val_ll,
            "test_log_loss": model_ll.mean(),
            "bet365_log_loss": b365_ll.mean(),
            "avg_log_loss": avg_ll.mean(),
        })

        print(
            f"{test_season} (train {train_seasons[0]}..{train_seasons[-1]}, val {val_season_list[0]}): "
            f"log loss train={train_ll:.4f} val={val_ll:.4f} test={model_ll.mean():.4f} "
            f"bet365={b365_ll.mean():.4f} avg={avg_ll.mean():.4f} "
            f"| acc model={model_correct.mean():.2%} bet365={b365_correct.mean():.2%} avg={avg_correct.mean():.2%}"
        )

    fold_df = pd.DataFrame(fold_rows)
    print("\n=== Per-fold summary (overfitting check: train vs. val vs. test) ===")
    print(fold_df[[
        "test_season", "train_log_loss", "val_log_loss", "test_log_loss", "bet365_log_loss", "avg_log_loss",
    ]].round(4))

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    avg_ll_all = np.concatenate(all_avg_ll)
    model_brier_all = np.concatenate(all_model_brier)
    b365_brier_all = np.concatenate(all_b365_brier)
    avg_brier_all = np.concatenate(all_avg_brier)
    model_correct_all = np.concatenate(all_model_correct)
    b365_correct_all = np.concatenate(all_b365_correct)
    avg_correct_all = np.concatenate(all_avg_correct)

    print(f"\n=== Pooled across all {len(fold_df)} folds ({len(model_ll_all)} test matches) ===")
    print(f"Model log loss:      {model_ll_all.mean():.4f}")
    print(f"Bet365 log loss:     {b365_ll_all.mean():.4f}")
    print(f"Avg-bookie log loss: {avg_ll_all.mean():.4f}")
    print(f"Model Brier:         {model_brier_all.mean():.4f}")
    print(f"Bet365 Brier:        {b365_brier_all.mean():.4f}")
    print(f"Avg-bookie Brier:    {avg_brier_all.mean():.4f}")
    print(f"Model accuracy (1X2):      {model_correct_all.mean():.2%}")
    print(f"Bet365 accuracy (1X2):     {b365_correct_all.mean():.2%}")
    print(f"Avg-bookie accuracy (1X2): {avg_correct_all.mean():.2%}")

    print(f"\nMean train log loss:      {fold_df['train_log_loss'].mean():.4f}")
    print(f"Mean validation log loss: {fold_df['val_log_loss'].mean():.4f}")
    print(
        f"Gap (val - train): {fold_df['val_log_loss'].mean() - fold_df['train_log_loss'].mean():.4f}  "
        "(large gap = overfitting)"
    )

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.reset_index(drop=True)
    print(f"Feature count: {len(FEATURES)}")
    walk_forward_vs_market(all_df)


if __name__ == "__main__":
    main()
