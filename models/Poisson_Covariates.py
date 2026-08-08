"""
Poisson.py's DixonColes model, but with covariate-driven scoring rates
(same 7 covariates as Poisson_Covariates_Bivariate.py: Elo, xG_Rolling5,
xGA_Rolling5, PPG, ShotOnTargetDifference_RollingTeam7) instead of per-team
attack/defense ratings. Home/away goals are still modeled as INDEPENDENT
Poisson counts, with the same low-score tau correction as Poisson.py --
this is the middle ground between plain Dixon-Coles (no covariates,
Poisson.py) and the full bivariate model (covariates + genuine correlation
via trivariate reduction, Poisson_Covariates_Bivariate.py).

Same intra-season cumulative sliding walk-forward as every other model
script (WINDOW=3, N_CHUNKS=5), on all_seasons_14window_ppg.csv.

RESULT: filled in after running -- see bottom of this docstring.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.preprocessing import label_binarize
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3
N_CHUNKS = 5

HOME_COVARIATES = [
    "Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_PPG",
    "Home_ShotOnTargetDifference_RollingTeam7", "Away_ShotOnTargetDifference_RollingTeam7",
]
AWAY_COVARIATES = [
    "Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_PPG",
    "Away_ShotOnTargetDifference_RollingTeam7", "Home_ShotOnTargetDifference_RollingTeam7",
]


class PoissonRegressionGoalsCovariates:

    def __init__(self):
        self.k = len(HOME_COVARIATES)
        self.beta_home = None
        self.beta_away = None
        self.home_adv = None
        self.rho = None
        self.home_mean = self.home_std = self.away_mean = self.away_std = None

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        """Dixon-Coles low-score correlation adjustment (only affects 0-0/0-1/1-0/1-1)."""
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

        fthg = matches_df["FTHG"].values.astype(float)
        ftag = matches_df["FTAG"].values.astype(float)
        k = self.k

        def neg_log_likelihood(params):
            beta_home = params[:k]
            beta_away = params[k:2 * k]
            home_adv = params[2 * k]
            rho = params[2 * k + 1]

            lam = np.exp(home_adv + X_home_std @ beta_home)
            mu = np.exp(X_away_std @ beta_away)

            ll = poisson.logpmf(fthg, lam) + poisson.logpmf(ftag, mu)
            tau_vals = np.array([
                self._tau(x, y, l, m, rho) for x, y, l, m in zip(fthg, ftag, lam, mu)
            ])
            tau_vals = np.clip(tau_vals, 1e-10, None)
            ll += np.log(tau_vals)
            return -ll.sum()

        x0 = np.zeros(2 * k + 2)
        x0[2 * k] = 0.2  # home advantage init

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

        self.beta_home = result.x[:k]
        self.beta_away = result.x[k:2 * k]
        self.home_adv = result.x[2 * k]
        self.rho = result.x[2 * k + 1]
        return self

    def predict_proba(self, matches_df, max_goals=10):
        """Returns an (n_matches, 3) array, columns ordered [H, D, A]."""
        X_home = matches_df[HOME_COVARIATES].fillna(0).values
        X_away = matches_df[AWAY_COVARIATES].fillna(0).values
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        lam_all = np.exp(self.home_adv + X_home_std @ self.beta_home)
        mu_all = np.exp(X_away_std @ self.beta_away)

        probs = []
        for lam, mu in zip(lam_all, mu_all):
            home_pmf = poisson.pmf(np.arange(max_goals + 1), lam)
            away_pmf = poisson.pmf(np.arange(max_goals + 1), mu)
            grid = np.outer(home_pmf, away_pmf)
            for x in range(2):
                for y in range(2):
                    grid[x, y] *= self._tau(x, y, lam, mu, self.rho)
            grid = grid / grid.sum()  # renormalize after the tau adjustment

            p_home = np.tril(grid, -1).sum()
            p_draw = np.trace(grid)
            p_away = np.triu(grid, 1).sum()
            probs.append((p_home, p_draw, p_away))
        return np.array(probs)


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
    seasons = sorted(all_df["Season"].unique())

    chunk_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct = []
    per_chunk_idx_ll = {i: [] for i in range(N_CHUNKS)}

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]

        prior_df = all_df[all_df["Season"].isin(train_seasons)]
        season_df = all_df[all_df["Season"] == test_season].sort_values("Date").reset_index(drop=True)
        chunks = np.array_split(season_df, N_CHUNKS)

        for c_idx, test_chunk in enumerate(chunks):
            if test_chunk.empty:
                continue
            elapsed_chunks = pd.concat(chunks[:c_idx]) if c_idx > 0 else season_df.iloc[0:0]
            train_df = pd.concat([prior_df, elapsed_chunks])

            model = PoissonRegressionGoalsCovariates().fit(train_df)

            proba = model.predict_proba(test_chunk)
            classes_order = ["H", "D", "A"]

            b365_overround = 1 / test_chunk["B365HomeOdds"] + 1 / test_chunk["B365DrawOdds"] + 1 / test_chunk["B365AwayOdds"]
            fair_b365 = np.column_stack([
                ((1 / test_chunk["B365HomeOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365DrawOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365AwayOdds"]) / b365_overround).values,
            ])

            avg_overround = 1 / test_chunk["AvgHomeOdds"] + 1 / test_chunk["AvgDrawOdds"] + 1 / test_chunk["AvgAwayOdds"]
            fair_avg = np.column_stack([
                ((1 / test_chunk["AvgHomeOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgDrawOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgAwayOdds"]) / avg_overround).values,
            ])

            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=classes_order)

            eps = 1e-15
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
            avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
            model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
            b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
            avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)
            model_pred = np.array(classes_order)[proba.argmax(axis=1)]
            model_correct = (model_pred == y_true)

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
                "model_ll": model_ll.mean(),
                "bet365_ll": b365_ll.mean(),
                "avg_ll": avg_ll.mean(),
            })

        print(f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)})")

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

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    walk_forward_vs_market(all_df)


if __name__ == "__main__":
    main()
