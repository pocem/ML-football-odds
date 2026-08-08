"""
Does PCA help RF/XGBoost? Compresses the full ~41-column feature set (same
drop_cols as the production RF/XGBoost cells) down to however many components
explain 95% of variance (StandardScaler + PCA fit on TRAIN only per fold, to
avoid leakage), then compares against the no-PCA baseline on identical folds.
Same WINDOW=3/VAL_SEASONS=1 walk-forward-vs-Bet365 structure as the production
notebooks, same hyperparameters already established for each model.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 3
VAL_SEASONS = 1

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


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, random_state=42, n_jobs=-1,
        eval_metric="mlogloss"
    )


def run(all_df, model_name, use_pca):
    le = LabelEncoder()
    le.fit(all_df["FTR"])
    validator = WalkForwardValidator(seasons=sorted(all_df["Season"].unique()))

    all_model_ll, all_b365_ll = [], []
    n_components_per_fold = []

    for train_idx, val_idx, test_idx, train_seasons, val_season_list, test_season in validator.split(
        all_df, window=WINDOW, val_seasons=VAL_SEASONS
    ):
        train_df = all_df.loc[train_idx]
        test_df = all_df.loc[test_idx]

        X_train = train_df.drop(columns=drop_cols).fillna(0)
        X_test = test_df.drop(columns=drop_cols).fillna(0)

        if model_name == "rf":
            y_train = train_df["FTR"]
        else:
            y_train = le.transform(train_df["FTR"])

        if use_pca:
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            pca = PCA(n_components=0.95, random_state=42).fit(X_train_s)
            X_train = pca.transform(X_train_s)
            X_test = pca.transform(X_test_s)
            n_components_per_fold.append(pca.n_components_)

        model = make_rf() if model_name == "rf" else make_xgb()
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        classes_order = model.classes_ if model_name == "rf" else le.classes_

        overround = 1 / test_df["B365HomeOdds"] + 1 / test_df["B365DrawOdds"] + 1 / test_df["B365AwayOdds"]
        fair = {
            "H": (1 / test_df["B365HomeOdds"]) / overround,
            "D": (1 / test_df["B365DrawOdds"]) / overround,
            "A": (1 / test_df["B365AwayOdds"]) / overround,
        }
        fair_market_proba = np.column_stack([fair[c].values for c in classes_order])

        y_true = test_df["FTR"].values
        y_true_for_onehot = y_true if model_name == "rf" else le.transform(y_true)
        onehot_classes = classes_order if model_name == "rf" else range(len(classes_order))
        y_onehot = label_binarize(y_true_for_onehot, classes=onehot_classes)

        eps = 1e-15
        model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
        b365_ll = -np.log(np.clip((fair_market_proba * y_onehot).sum(axis=1), eps, 1))

        all_model_ll.append(model_ll)
        all_b365_ll.append(b365_ll)

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    tag = "PCA(95% var)" if use_pca else "no PCA (full features)"
    extra = f" avg_n_components={np.mean(n_components_per_fold):.1f}" if use_pca else f" n_features={X_train.shape[1] if not use_pca else ''}"
    print(f"{model_name.upper():4s} {tag:24s} model_ll={model_ll_all.mean():.4f}  bet365_ll={b365_ll_all.mean():.4f}{extra if use_pca else ''}")


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.reset_index(drop=True)
    print(f"Full feature count (pre-PCA): {all_df.drop(columns=drop_cols).shape[1]}")
    for model_name in ["rf", "xgb"]:
        for use_pca in [False, True]:
            run(all_df, model_name, use_pca)


if __name__ == "__main__":
    main()
