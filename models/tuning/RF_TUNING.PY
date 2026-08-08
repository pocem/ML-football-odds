"""
Random Forest hyperparameter tuning via RandomizedSearchCV, scored on log
loss (i.e. via predict_proba(), not hard accuracy) using walk-forward
cross-validation over seasons -- consistent with the rest of this project.

The most recent season (FINAL_TEST_SEASON) is never touched during the
search; it's held out purely for the final model-vs-market comparison, so
the reported test numbers aren't contaminated by hyperparameter selection.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize

# -------------------------------------------------------------------------
# Config -- change values here only, nothing is re-hardcoded elsewhere.
# -------------------------------------------------------------------------
DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 4              # seasons of history per walk-forward CV fold
FINAL_TEST_SEASON = "24-25"
N_ITER = 60             # number of randomized hyperparameter combos to try
RANDOM_STATE = 42

DROP_COLS = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies"
]

# The train/val gap we measured earlier (0.65 vs ~1.0 log loss) came from
# max_depth=10 with only min_samples_leaf=5 -- this search leans the range
# toward shallower, more constrained trees to see if that gap can close.
# NOTE: max_depth deliberately has no "None" (unlimited) option -- we already
# have strong evidence in this project that unlimited depth overfits badly,
# and letting noisy CV "rediscover" that the hard way just wastes search
# budget on candidates we don't want anyway.
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_leaf": [1, 2, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
}


def build_walk_forward_splits(meta, window):
    """Positional (train_idx, test_idx) pairs for RandomizedSearchCV's `cv`."""
    seasons = sorted(meta["Season"].unique())
    splits = []
    for i in range(window, len(seasons)):
        train_seasons = seasons[i - window:i]
        test_season = seasons[i]
        train_idx = meta.index[meta["Season"].isin(train_seasons)].to_numpy()
        test_idx = meta.index[meta["Season"] == test_season].to_numpy()
        splits.append((train_idx, test_idx))
    return splits


def rf_complexity(params):
    """Lower = simpler/more regularized. Depth dominates (that's what drove
    the overfitting we measured), min_samples_leaf is the tiebreaker."""
    depth = params["max_depth"] if params["max_depth"] is not None else 1000
    return (depth, -params["min_samples_leaf"])


def select_simplest_within_one_se(search, n_folds, complexity_key):
    """
    RandomizedSearchCV.best_params_ just takes the single highest-scoring
    candidate -- on noisy CV folds (football's per-match variance is huge,
    we measured it), an unregularized/overfit candidate can win that by pure
    chance without actually generalizing. This picks the SIMPLEST candidate
    whose mean CV score is within one standard error of the best instead --
    the standard defense against overfitting to the CV score itself.
    """
    results = pd.DataFrame(search.cv_results_)
    best_score = results["mean_test_score"].max()
    best_idx = results["mean_test_score"].idxmax()
    best_se = results.loc[best_idx, "std_test_score"] / np.sqrt(n_folds)
    threshold = best_score - best_se

    within_1se = results[results["mean_test_score"] >= threshold].copy()
    within_1se["complexity"] = within_1se["params"].apply(complexity_key)
    best_row_idx = min(within_1se.index, key=lambda idx: within_1se.loc[idx, "complexity"])
    chosen = within_1se.loc[best_row_idx]

    print(f"Naive best CV log loss:  {-best_score:.4f}  params: {results.loc[best_idx, 'params']}")
    print(f"1-SE threshold: log loss <= {-threshold:.4f}  ({len(within_1se)}/{len(results)} candidates qualify)")
    print(f"Chosen (simplest within 1 SE): log loss={-chosen['mean_test_score']:.4f}  params: {chosen['params']}")

    return chosen["params"]


def main():
    all_df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    unique_seasons = sorted(all_df["Season"].unique())

    # Only use seasons up to and including FINAL_TEST_SEASON -- otherwise a
    # later season (e.g. 25-26, if testing on an earlier one) would leak into
    # the CV folds used for tuning even though it's never the actual target.
    usable_seasons = unique_seasons[:unique_seasons.index(FINAL_TEST_SEASON) + 1]
    all_df = all_df[all_df["Season"].isin(usable_seasons)].reset_index(drop=True)
    unique_seasons = usable_seasons

    # --- Held out completely from tuning ---
    dev_df = all_df[all_df["Season"] != FINAL_TEST_SEASON].reset_index(drop=True)

    cv_splits = build_walk_forward_splits(dev_df, WINDOW)
    print(f"Walk-forward CV: {len(cv_splits)} folds, window={WINDOW} seasons")

    X_dev = dev_df.drop(columns=DROP_COLS).fillna(0)
    y_dev = dev_df["FTR"]

    # n_jobs=1 on the estimator itself -- parallelism happens across the
    # (candidate x fold) grid via the search's own n_jobs=-1 instead, which
    # uses many small fits more efficiently than nested parallelism would.
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced")

    search = RandomizedSearchCV(
        rf,
        PARAM_DIST,
        n_iter=N_ITER,
        scoring="neg_log_loss",
        cv=cv_splits,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=False,
        verbose=1,
    )

    print(f"\nSearching {N_ITER} candidates x {len(cv_splits)} folds = {N_ITER * len(cv_splits)} fits...")
    t0 = time.time()
    search.fit(X_dev, y_dev)
    print(f"Search took {time.time() - t0:.1f}s\n")

    chosen_params = select_simplest_within_one_se(search, len(cv_splits), rf_complexity)

    # -------------------------------------------------------------------
    # Final model: refit with the chosen params on exactly the last WINDOW
    # seasons before the held-out test season (same recency window the
    # rest of this project uses), then score it against the market.
    # -------------------------------------------------------------------
    final_train_seasons = unique_seasons[
        unique_seasons.index(FINAL_TEST_SEASON) - WINDOW: unique_seasons.index(FINAL_TEST_SEASON)
    ]
    print(f"\nFinal train seasons: {final_train_seasons}")
    print(f"Final test season: {FINAL_TEST_SEASON}")

    train_df = all_df[all_df["Season"].isin(final_train_seasons)]
    test_df = all_df[all_df["Season"] == FINAL_TEST_SEASON].copy()

    X_train = train_df.drop(columns=DROP_COLS).fillna(0)
    y_train = train_df["FTR"]
    X_test = test_df.drop(columns=DROP_COLS).fillna(0)

    final_rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **chosen_params
    )
    final_rf.fit(X_train, y_train)

    # --- Overfitting check: train vs. test log loss for the tuned model ---
    train_proba = final_rf.predict_proba(X_train)
    train_ll = log_loss(y_train, train_proba, labels=final_rf.classes_)

    proba = final_rf.predict_proba(X_test)
    classes_order = final_rf.classes_

    # De-vig the market odds so it's a fair fight against the model's probabilities.
    overround = 1 / test_df["AvgHomeOdds"] + 1 / test_df["AvgDrawOdds"] + 1 / test_df["AvgAwayOdds"]
    fair = {
        "H": (1 / test_df["AvgHomeOdds"]) / overround,
        "D": (1 / test_df["AvgDrawOdds"]) / overround,
        "A": (1 / test_df["AvgAwayOdds"]) / overround,
    }
    fair_market_proba = np.column_stack([fair[c].values for c in classes_order])

    y_true = test_df["FTR"].values
    y_onehot = label_binarize(y_true, classes=classes_order)

    model_ll = log_loss(y_true, proba, labels=classes_order)
    market_ll = log_loss(y_true, fair_market_proba, labels=classes_order)
    model_brier = ((proba - y_onehot) ** 2).sum(axis=1).mean()
    market_brier = ((fair_market_proba - y_onehot) ** 2).sum(axis=1).mean()

    print(f"\n=== Overfitting check ===")
    print(f"Train log loss: {train_ll:.4f}")
    print(f"Test log loss:  {model_ll:.4f}  (gap: {model_ll - train_ll:.4f})")

    print(f"\n=== Final model vs. market on {FINAL_TEST_SEASON} ({len(test_df)} matches) ===")
    print(f"Model log loss:  {model_ll:.4f}")
    print(f"Market log loss: {market_ll:.4f}")
    print(f"Model Brier:     {model_brier:.4f}")
    print(f"Market Brier:    {market_brier:.4f}")


if __name__ == "__main__":
    main()
