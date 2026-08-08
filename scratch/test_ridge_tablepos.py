import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.preprocessing import label_binarize

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 3
LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

HOME_COVARIATES = ["Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_TablePosDiff"]
AWAY_COVARIATES = ["Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_TablePosDiff"]


class PoissonRegressionGoalsRidge:
    def __init__(self, lam=LAMBDA):
        self.lam = lam
        self.k = len(HOME_COVARIATES)
        self.beta_home = None
        self.beta_away = None
        self.home_adv = None
        self.rho = None
        self.home_mean = self.home_std = self.away_mean = self.away_std = None

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, matches_df):
        X_home = matches_df[HOME_COVARIATES].fillna(0).values
        X_away = matches_df[AWAY_COVARIATES].fillna(0).values
        self.home_mean, self.home_std = X_home.mean(axis=0), X_home.std(axis=0)
        self.away_mean, self.away_std = X_away.mean(axis=0), X_away.std(axis=0)
        self.home_std[self.home_std == 0] = 1
        self.away_std[self.away_std == 0] = 1
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        fthg = matches_df["FTHG"].values
        ftag = matches_df["FTAG"].values
        k = self.k

        def neg_log_likelihood(params):
            beta_home = params[:k]
            beta_away = params[k:2 * k]
            home_adv = params[2 * k]
            rho = params[2 * k + 1]

            lam_ = np.exp(home_adv + X_home_std @ beta_home)
            mu_ = np.exp(X_away_std @ beta_away)

            ll = poisson.logpmf(fthg, lam_) + poisson.logpmf(ftag, mu_)
            tau_vals = np.array([
                self._tau(x, y, l, m, rho) for x, y, l, m in zip(fthg, ftag, lam_, mu_)
            ])
            tau_vals = np.clip(tau_vals, 1e-10, None)
            ll += np.log(tau_vals)
            nll = -ll.sum()
            # Ridge penalty only on the TablePosDiff coefficients (last index of each vector)
            nll += self.lam * (beta_home[-1] ** 2 + beta_away[-1] ** 2)
            return nll

        x0 = np.zeros(2 * k + 2)
        x0[2 * k] = 0.2
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        self.beta_home = result.x[:k]
        self.beta_away = result.x[k:2 * k]
        self.home_adv = result.x[2 * k]
        self.rho = result.x[2 * k + 1]
        return self

    def predict_proba(self, matches_df, max_goals=10):
        X_home = matches_df[HOME_COVARIATES].fillna(0).values
        X_away = matches_df[AWAY_COVARIATES].fillna(0).values
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std
        lam_all = np.exp(self.home_adv + X_home_std @ self.beta_home)
        mu_all = np.exp(X_away_std @ self.beta_away)

        probs = []
        for lam_, mu_ in zip(lam_all, mu_all):
            home_pmf = poisson.pmf(np.arange(max_goals + 1), lam_)
            away_pmf = poisson.pmf(np.arange(max_goals + 1), mu_)
            grid = np.outer(home_pmf, away_pmf)
            for x in range(2):
                for y in range(2):
                    grid[x, y] *= self._tau(x, y, lam_, mu_, self.rho)
            grid = grid / grid.sum()
            probs.append((np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()))
        return np.array(probs)


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    seasons = sorted(all_df["Season"].unique())

    all_model_ll, all_b365_ll = [], []
    all_model_correct, all_b365_correct = [], []
    tablepos_coefs = []

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]
        train_df = all_df[all_df["Season"].isin(train_seasons)]
        test_df = all_df[all_df["Season"] == test_season]

        model = PoissonRegressionGoalsRidge().fit(train_df)
        tablepos_coefs.append((model.beta_home[-1], model.beta_away[-1]))

        proba = model.predict_proba(test_df)
        classes_order = ["H", "D", "A"]

        overround = 1 / test_df["B365HomeOdds"] + 1 / test_df["B365DrawOdds"] + 1 / test_df["B365AwayOdds"]
        fair_h = (1 / test_df["B365HomeOdds"]) / overround
        fair_d = (1 / test_df["B365DrawOdds"]) / overround
        fair_a = (1 / test_df["B365AwayOdds"]) / overround
        fair_market_proba = np.column_stack([fair_h.values, fair_d.values, fair_a.values])

        y_true = test_df["FTR"].values
        y_onehot = label_binarize(y_true, classes=classes_order)

        eps = 1e-15
        model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
        b365_ll = -np.log(np.clip((fair_market_proba * y_onehot).sum(axis=1), eps, 1))
        model_pred = np.array(classes_order)[proba.argmax(axis=1)]
        b365_pred = np.array(classes_order)[fair_market_proba.argmax(axis=1)]

        all_model_ll.append(model_ll)
        all_b365_ll.append(b365_ll)
        all_model_correct.append(model_pred == y_true)
        all_b365_correct.append(b365_pred == y_true)

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    model_correct_all = np.concatenate(all_model_correct)
    b365_correct_all = np.concatenate(all_b365_correct)
    n = len(model_ll_all)
    model_wins = (model_ll_all < b365_ll_all).sum()

    avg_home_coef = np.mean([c[0] for c in tablepos_coefs])
    avg_away_coef = np.mean([c[1] for c in tablepos_coefs])

    print(
        f"LAMBDA={LAMBDA:<6} model_ll={model_ll_all.mean():.4f} "
        f"model_acc={model_correct_all.mean():.4f} winrate={model_wins/n:.4f} "
        f"avg_tablepos_coef(home,away)=({avg_home_coef:.4f},{avg_away_coef:.4f})"
    )


if __name__ == "__main__":
    main()
