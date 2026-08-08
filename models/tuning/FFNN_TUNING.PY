"""
FFNN (MLPClassifier) hyperparameter tuning via RandomizedSearchCV, scored on
log loss (i.e. via predict_proba(), not hard accuracy) using walk-forward
cross-validation over seasons -- same structure as LR_TUNING.PY/RF_TUNING.PY/
XGB_TUNING.PY.

Search space: hidden_layer_sizes (a handful of small architectures -- see
models/FFNN.py's docstring for why small matters here: even a single 16-unit
hidden layer is already ~1,500 weights for ~700-1400 training rows) and alpha
(L2 penalty, log-uniform).

Early stopping nuance: during the RandomizedSearchCV phase, each candidate
uses sklearn's built-in early_stopping=True (a random validation_fraction
carved out of that CV fold's training data) -- not the season-aware manual
early stopping FFNN.py uses, since wiring a custom season-based eval set
through RandomizedSearchCV's own CV splitting isn't practical. That's fine
for RANKING candidates against each other during the search; the FINAL
refit below switches to the same manual, season-based early stopping as
FFNN.py, so the reported final result is directly comparable to it.

The most recent season (FINAL_TEST_SEASON) is never touched during the
search; it's held out purely for the final model-vs-market comparison, so
the reported test numbers aren't contaminated by hyperparameter selection.
"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# -------------------------------------------------------------------------
# Config -- change values here only, nothing is re-hardcoded elsewhere.
# -------------------------------------------------------------------------
DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 4              # seasons of history per walk-forward CV fold
FINAL_TEST_SEASON = "24-25"
N_ITER = 30             # number of randomized hyperparameter combos to try
RANDOM_STATE = 42

DROP_COLS = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
]

PARAM_DIST = {
    "mlp__hidden_layer_sizes": [(8,), (16,), (32,), (8, 8), (16, 8)],
    "mlp__alpha": loguniform(1e-4, 1e1),
}

# Final-refit early stopping (same as models/FFNN.py's manual loop)
MAX_EPOCHS = 300
PATIENCE = 15


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


def nn_complexity(params, n_features, n_classes=3):
    """Lower = simpler/more regularized. Total weight count is the primary
    complexity measure (more weights = more capacity); among architectures
    of similar size, stronger regularization (larger alpha) is simpler."""
    hidden_sizes = params["mlp__hidden_layer_sizes"]
    layer_sizes = [n_features] + list(hidden_sizes) + [n_classes]
    n_weights = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
    return (n_weights, -params["mlp__alpha"])


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


def eval_ll(model, X, y, classes_order):
    proba = model.predict_proba(X)
    y_onehot = label_binarize(y, classes=classes_order)
    eps = 1e-15
    return -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1)).mean()


def train_with_early_stopping(X_train, y_train, X_val, y_val, classes_order, hidden_layer_sizes, alpha):
    """Same manual epoch-by-epoch early stopping as FFNN.py, using the true
    held-out validation season rather than RandomizedSearchCV's internal
    early_stopping=True (which only used a random split, fine for ranking
    candidates during the search but not for a final, comparable result)."""
    nn = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
    )

    best_val_ll = np.inf
    best_model = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        nn.partial_fit(X_train, y_train, classes=classes_order)
        val_ll = eval_ll(nn, X_val, y_val, classes_order)

        if val_ll < best_val_ll - 1e-5:
            best_val_ll = val_ll
            best_model = deepcopy(nn)
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    return best_model, best_epoch


def main():
    all_df = pd.read_csv(DATA_PATH).reset_index(drop=True)
    unique_seasons = sorted(all_df["Season"].unique())

    # Only use seasons up to and including FINAL_TEST_SEASON -- otherwise a
    # later season (e.g. 25-26) would leak into the CV folds used for tuning
    # even though it's never the actual target.
    usable_seasons = unique_seasons[:unique_seasons.index(FINAL_TEST_SEASON) + 1]
    all_df = all_df[all_df["Season"].isin(usable_seasons)].reset_index(drop=True)
    unique_seasons = usable_seasons

    # --- Held out completely from tuning ---
    dev_df = all_df[all_df["Season"] != FINAL_TEST_SEASON].reset_index(drop=True)

    cv_splits = build_walk_forward_splits(dev_df, WINDOW)
    print(f"Walk-forward CV: {len(cv_splits)} folds, window={WINDOW} seasons")

    X_dev = dev_df.drop(columns=DROP_COLS).fillna(0)
    # Integer-encode y: MLPClassifier's built-in early_stopping=True computes its
    # internal validation score via a path that calls np.isnan() directly on the
    # predicted labels, which errors out on string labels ('H'/'D'/'A'). Only
    # needed for this search phase -- the final refit's manual partial_fit loop
    # doesn't hit that code path and works fine with string labels.
    label_encoder = LabelEncoder()
    y_dev = label_encoder.fit_transform(dev_df["FTR"])
    n_features = X_dev.shape[1]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,   # random-split early stopping -- fine for ranking candidates
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )),
    ])

    search = RandomizedSearchCV(
        pipe,
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

    chosen_params = select_simplest_within_one_se(
        search, len(cv_splits), lambda p: nn_complexity(p, n_features)
    )

    # -------------------------------------------------------------------
    # Final model: refit with the chosen params on the last WINDOW seasons
    # before the held-out test season, with ONE of those seasons carved out
    # as the validation season for manual, season-based early stopping (same
    # scheme as FFNN.py) -- not the random-split early stopping used during
    # the search above, so this result is directly comparable to FFNN.py's.
    # -------------------------------------------------------------------
    final_window_seasons = unique_seasons[
        unique_seasons.index(FINAL_TEST_SEASON) - WINDOW: unique_seasons.index(FINAL_TEST_SEASON)
    ]
    final_train_seasons = final_window_seasons[:-1]
    final_val_season = final_window_seasons[-1]
    print(f"\nFinal train seasons: {final_train_seasons}")
    print(f"Final validation season (early stopping): {final_val_season}")
    print(f"Final test season: {FINAL_TEST_SEASON}")

    train_df = all_df[all_df["Season"].isin(final_train_seasons)]
    val_df = all_df[all_df["Season"] == final_val_season]
    test_df = all_df[all_df["Season"] == FINAL_TEST_SEASON].copy()

    X_train_raw = train_df.drop(columns=DROP_COLS).fillna(0)
    y_train = train_df["FTR"]
    X_val_raw = val_df.drop(columns=DROP_COLS).fillna(0)
    y_val = val_df["FTR"]
    X_test_raw = test_df.drop(columns=DROP_COLS).fillna(0)

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    hidden_layer_sizes = chosen_params["mlp__hidden_layer_sizes"]
    alpha = chosen_params["mlp__alpha"]
    classes_order = np.array(sorted(y_train.unique()))

    final_model, best_epoch = train_with_early_stopping(
        X_train, y_train, X_val, y_val, classes_order, hidden_layer_sizes, alpha
    )
    print(f"Final model stopped at epoch {best_epoch} (max {MAX_EPOCHS})")

    proba = final_model.predict_proba(X_test)
    classes_order = final_model.classes_

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

    print(f"\n=== Final model vs. market on {FINAL_TEST_SEASON} ({len(test_df)} matches) ===")
    print(f"Model log loss:  {model_ll:.4f}")
    print(f"Market log loss: {market_ll:.4f}")
    print(f"Model Brier:     {model_brier:.4f}")
    print(f"Market Brier:    {market_brier:.4f}")

    print(f"\nChosen hyperparameters: hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha:.6f}")
    print("Update models/FFNN.py's HIDDEN_LAYER_SIZES/ALPHA with these values to deploy.")


if __name__ == "__main__":
    main()
