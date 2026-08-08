import sys
sys.path.insert(0, r"c:\Users\misog\SCHOOL\Summer project\ML-football-odds\models")
import numpy as np
import pandas as pd
from Poisson_Covariates_Bivariate import PoissonRegressionGoalsBivariate, DATA_PATH
from sklearn.preprocessing import label_binarize

WINDOWS = [2, 3, 4, 5, 6, 7]


def run(all_df, window):
    seasons = sorted(all_df["Season"].unique())
    all_model_ll, all_model_correct = [], []

    for i in range(window, len(seasons)):
        train_seasons = seasons[i - window:i]
        test_season = seasons[i]
        train_df = all_df[all_df["Season"].isin(train_seasons)]
        test_df = all_df[all_df["Season"] == test_season]

        model = PoissonRegressionGoalsBivariate().fit(train_df)
        proba = model.predict_proba(test_df)
        classes_order = ["H", "D", "A"]

        y_true = test_df["FTR"].values
        y_onehot = label_binarize(y_true, classes=classes_order)
        eps = 1e-15
        model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
        model_pred = np.array(classes_order)[proba.argmax(axis=1)]

        all_model_ll.append(model_ll)
        all_model_correct.append(model_pred == y_true)
        print(f"  {test_season}: log_loss={model_ll.mean():.4f} acc={(model_pred == y_true).mean():.2%}")

    model_ll_all = np.concatenate(all_model_ll)
    model_correct_all = np.concatenate(all_model_correct)
    n_folds = len(seasons) - window
    print(
        f"WINDOW={window:<2} n_folds={n_folds:<2} n_matches={len(model_ll_all):<5} "
        f"log_loss={model_ll_all.mean():.4f} acc={model_correct_all.mean():.2%}"
    )


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    for w in WINDOWS:
        run(all_df, w)


if __name__ == "__main__":
    main()
