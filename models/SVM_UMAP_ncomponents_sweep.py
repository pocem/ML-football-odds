"""
Sweeps UMAP's N_components (the dimensionality-reduction target in
SVM_UMAP.py), starting at 10 and stepping by 2, running until convergence
rather than a fixed list -- same convergence rule as
Poisson_Bivariate_intraseason_rolling_window_sweep.py's EWMA span sweep
elsewhere in this project: track the running minimum validation log loss;
if PATIENCE consecutive steps pass without a new minimum, the sweep is
done -- run EXTRA_POINTS more points past that, then stop.

Uses the same WINDOW=3 / VAL_SEASONS=1 walk-forward split as SVM_UMAP.py, but
reports TRAIN vs VALIDATION log loss, not test-vs-Bet365 -- this is a
hyperparameter-selection exercise, so it's the held-out validation season
inside each fold that should drive the choice of N_components, keeping the
test folds untouched by tuning (same discipline as every other tuning script
in this project).

Produces one plot: mean train log loss and mean validation log loss (averaged
across all walk-forward folds) vs. N_components, saved to
images/umap_ncomponents_sweep.png.

Note: one random_state per N_components, matching every other script in this
project -- UMAP's fit is a stochastic manifold optimization (unlike PCA's
closed-form one), so the curve could be a little noisy from seed variance
alone. Treat it as indicative of the general trend, not an exact optimum.

This script is compute-heavy (UMAP + SVC fit from scratch for every one of 9
folds x every N_components value tried) -- expect a real run time, not a
quick one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from umap import UMAP

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3
VAL_SEASONS = 1
START_N_COMPONENTS = 10
STEP = 2
PATIENCE = 3       # consecutive non-improving steps before declaring convergence
EXTRA_POINTS = 2   # additional steps to run past the convergence point
MAX_N_COMPONENTS = 40  # safety cap
OUTPUT_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\images\umap_ncomponents_sweep.png"

drop_cols = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
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
    """Log loss for a fitted model against a labeled set."""
    proba = model.predict_proba(X)
    y_onehot = label_binarize(y, classes=classes_order)
    eps = 1e-15
    ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1)).mean()
    return ll


def run_for_n_components(all_df, n_components):
    validator = WalkForwardValidator(seasons=sorted(all_df["Season"].unique()))
    fold_train_ll, fold_val_ll = [], []

    for train_idx, val_idx, test_idx, train_seasons, val_season_list, test_season in validator.split(
        all_df, window=WINDOW, val_seasons=VAL_SEASONS
    ):
        train_df = all_df.loc[train_idx]
        val_df = all_df.loc[val_idx]

        X_train_raw = train_df.drop(columns=drop_cols).fillna(0)
        y_train = train_df["FTR"]
        X_val_raw = val_df.drop(columns=drop_cols).fillna(0)
        y_val = val_df["FTR"]

        scaler = StandardScaler().fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        reducer = UMAP(n_components=n_components, random_state=42)
        X_train = reducer.fit_transform(X_train_scaled)
        X_val = reducer.transform(X_val_scaled)

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

        fold_train_ll.append(eval_probs(svm, X_train, y_train, classes_order))
        fold_val_ll.append(eval_probs(svm, X_val, y_val, classes_order))

    return float(np.mean(fold_train_ll)), float(np.mean(fold_val_ll))


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.reset_index(drop=True)

    results = []
    best_val_ll = np.inf
    steps_since_improve = 0
    extra_run = 0
    n = START_N_COMPONENTS

    while True:
        train_ll, val_ll = run_for_n_components(all_df, n)
        print(f"N_components={n:2d}  train_ll={train_ll:.4f}  val_ll={val_ll:.4f}  gap={val_ll - train_ll:.4f}")
        results.append({"n_components": n, "train_ll": train_ll, "val_ll": val_ll})

        if val_ll < best_val_ll - 1e-5:
            best_val_ll = val_ll
            steps_since_improve = 0
        else:
            steps_since_improve += 1

        if steps_since_improve >= PATIENCE:
            extra_run += 1
            if extra_run > EXTRA_POINTS:
                print(f"\nConverged: no improvement for {PATIENCE} steps, ran {EXTRA_POINTS} extra points past that.")
                break

        n += STEP
        if n > MAX_N_COMPONENTS:
            print(f"\nHit MAX_N_COMPONENTS={MAX_N_COMPONENTS} safety cap, stopping.")
            break

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["val_ll"].idxmin()]
    print(
        f"\nBest N_components by validation log loss: {int(best_row['n_components'])} "
        f"(val_ll={best_row['val_ll']:.4f})"
    )

    n_components_list = results_df["n_components"].tolist()

    plt.figure(figsize=(9, 6))
    plt.plot(results_df["n_components"], results_df["train_ll"], marker="o", label="Train log loss")
    plt.plot(results_df["n_components"], results_df["val_ll"], marker="o", label="Validation log loss")
    plt.axvline(
        best_row["n_components"], color="gray", linestyle="--", alpha=0.6,
        label=f"Best N_components = {int(best_row['n_components'])}"
    )
    plt.xlabel("UMAP N_components")
    plt.ylabel("Log loss (mean across walk-forward folds)")
    plt.title("SVM + UMAP: train vs. validation log loss by N_components")
    plt.xticks(n_components_list)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
