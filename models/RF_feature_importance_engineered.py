"""
Feature importance check on the full ENGINEERED dataset
(dataset/all_seasons_14window_ppg.csv), unlike RF_feature_importance_raw_stats.py
which looked at raw same-match in-play stats (explanatory only, not usable
pre-match). Every column here IS legitimate pre-match information: rolling
EWMA stats (span=14, computed only from past matches), Elo ratings, and PPG
(both carry only information available before kickoff). Bookmaker odds and
identifiers/target are excluded the same way every model script in this
project excludes them.

Two importance measures are reported side by side, since they can disagree
and disagreement is itself informative:

- Gini importance (rf.feature_importances_): fit on ALL matches pooled
  together, no held-out set. Cheap, but biased toward high-cardinality/
  high-variance features and only reflects training-set splits, not
  generalization -- same caveat as before, just less severe here since these
  features aren't circular/contemporaneous with the match being predicted.
- Permutation importance: fit on seasons up to 24-25, evaluated by shuffling
  each feature on the held-out 25-26 season and measuring the drop in
  log loss. This one actually measures each feature's contribution to
  out-of-sample forecasting, at the cost of being noisier (single train/test
  split, not walk-forward) and slower (n_repeats reshuffles per feature).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
TOP_N = 15
HOLDOUT_SEASON = "25-26"  # most recent season, held out for permutation importance
N_REPEATS = 20
RANDOM_STATE = 42

drop_cols = [
    "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season",
    "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds", "NumBookies", "B365HomeOdds", "B365DrawOdds", "B365AwayOdds"
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=drop_cols).fillna(0)
    y = df["FTR"]
    print(f"{df.shape[0]} matches, {X.shape[1]} candidate features")

    # --- Gini importance: fit on everything pooled ---
    rf_full = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_full.fit(X, y)

    gini_importances = pd.Series(rf_full.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n=== Top {TOP_N} Gini (feature_importances_) -- fit on all {len(X)} matches pooled ===")
    print(gini_importances.head(TOP_N).round(4).to_string())
    print(f"\nTrain accuracy (sanity check, not a forecasting metric): {rf_full.score(X, y):.4f}")

    # --- Permutation importance: fit on train seasons, evaluate on held-out season ---
    train_mask = df["Season"] != HOLDOUT_SEASON
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"\nPermutation importance split: {len(X_train)} train ({df.loc[train_mask, 'Season'].nunique()} seasons) "
          f"/ {len(X_test)} test ({HOLDOUT_SEASON})")

    rf_holdout = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_holdout.fit(X_train, y_train)
    holdout_ll = log_loss(y_test, rf_holdout.predict_proba(X_test), labels=rf_holdout.classes_)
    print(f"Held-out log loss: {holdout_ll:.4f}  |  held-out accuracy: {rf_holdout.score(X_test, y_test):.4f}")

    perm = permutation_importance(
        rf_holdout, X_test, y_test,
        scoring="neg_log_loss",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    perm_std = pd.Series(perm.importances_std, index=X.columns)

    print(f"\n=== Top {TOP_N} permutation importance (mean log-loss increase when shuffled, held-out {HOLDOUT_SEASON}) ===")
    for name in perm_importances.head(TOP_N).index:
        print(f"{name:40s}  {perm_importances[name]:+.4f}  (+/- {perm_std[name]:.4f})")

    # --- Side-by-side comparison for the union of both top-N lists ---
    top_union = sorted(set(gini_importances.head(TOP_N).index) | set(perm_importances.head(TOP_N).index))
    comparison = pd.DataFrame({
        "gini_importance": gini_importances.reindex(top_union),
        "gini_rank": gini_importances.rank(ascending=False).reindex(top_union),
        "perm_importance": perm_importances.reindex(top_union),
        "perm_rank": perm_importances.rank(ascending=False).reindex(top_union),
    }).sort_values("perm_rank")
    print(f"\n=== Union of both top-{TOP_N} lists, side by side ===")
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    main()
