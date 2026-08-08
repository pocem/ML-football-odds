"""
Feature importance check on the ORIGINAL, raw per-season CSVs (pl21-22.csv
through pl24-25.csv, i.e. "pl21-25") -- not the engineered
all_seasons_with_bookies.csv. Random Forest fit directly on the raw in-match
stats to predict FTR, purely as an explanatory/retrospective diagnostic: this
is NOT a forecasting model (HS/AS/HST/AST/etc. are contemporaneous with the
match itself, generated during play -- using them to predict that same
match's FTR is circular, not real prediction). The point is to see which raw
stats have a strong relationship with the outcome at all, as evidence for
which ones were worth turning into PRE-match rolling-average features
elsewhere in this project, and which weren't.

Features used (matches process_season_data.py's original keep_columns,
minus identifiers/target, and minus HTHG/HTAG/HTR): HS, AS, HST, AST, HF,
AF, HC, AC, HY, AY, HR, AR. Half-time score/result was dropped too -- it's
an even more direct leak than the other stats, since it reveals a large
chunk of the actual outcome trajectory of the very match being predicted
(the match is half over), not just a loosely-correlated stat. Bookmaker
odds columns (B365*, BW*, BF*, PS*, WH*, 1XB*, Max*, Avg*, BFE*, and all
their closing/over-under/Asian-handicap variants) are excluded by using an
explicit allow-list rather than a drop-list, since the raw files have
dozens of odds columns across many bookmakers and markets -- an allow-list
is safer than trying to enumerate every one to exclude.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEASON_FILES = ["pl21-22.csv", "pl22-23.csv", "pl23-24.csv", "pl24-25.csv"]

FEATURE_COLS = [
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
]


def main():
    frames = [pd.read_csv(f, usecols=FEATURE_COLS + ["FTR"]) for f in SEASON_FILES]
    df = pd.concat(frames, ignore_index=True).dropna()
    print(f"Pooled pl21-22..pl24-25: {df.shape[0]} matches")

    X = df[FEATURE_COLS].copy()
    y = df["FTR"]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n=== Feature importances (raw in-match stats -> FTR, same-match, explanatory only) ===")
    print(importances.round(4).to_string())

    print(f"\nTrain accuracy (sanity check, not a forecasting metric): {rf.score(X, y):.4f}")


if __name__ == "__main__":
    main()
