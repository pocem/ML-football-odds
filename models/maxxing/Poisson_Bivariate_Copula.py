"""
Both requested fixes to the best-performing model (bivariate Poisson with
covariates), combined:

  1. lambda3 >= 0 non-negative-correlation constraint -- Poisson_Covariates_
     Bivariate.py's trivariate-reduction construction can only ever produce
     a non-negative correlation between home/away goals (lambda3 = e^theta
     is a shared Poisson component, which can only ADD covariance, never
     subtract it). Replaced here with a Gaussian copula joining two
     independent Poisson marginals, whose correlation parameter
     rho = tanh(z) is unconstrained in sign.

  2. New-team covariate handling -- Poisson_Covariates_Bivariate.py's
     0-fill-then-standardize maps a missing covariate to a wild outlier
     (e.g. ~-14.8 SD for Elo). Replaced with mean-imputation: missing
     values are filled with the TRAINING SET's own per-column mean
     (skipping NaNs) before standardizing, so a missing value maps to
     exactly 0 in standardized space -- "no information yet, assume
     league average" (same fix isolated and verified separately in
     Poisson_Bivariate_MeanImpute.py).

Same 7 covariates, same intra-season cumulative sliding walk-forward
(WINDOW=3, N_CHUNKS=5) as every other model in this project.

Gaussian copula construction: for discrete marginals, the standard
"rectangle" formula is
    P(X=x, Y=y) = C(F1(x),F2(y)) - C(F1(x-1),F2(y))
                  - C(F1(x),F2(y-1)) + C(F1(x-1),F2(y-1))
where F1/F2 are the two (independent) Poisson marginal CDFs and C is the
bivariate standard normal CDF at correlation rho, evaluated at each
marginal's normal-quantile transform (Sklar's theorem). F(-1) := 0 for the
x=0/y=0 boundary case -- implemented by clipping those CDF values to a
small epsilon before the normal-quantile transform, rather than passing
literal -inf into the bivariate normal CDF.

RESULT (45-fold walk-forward on data/processed/all_seasons_14window_ppg.csv,
recorded when this was first run): fitted rho ranged -0.086 to +0.058
(mean -0.018), negative on 34/45 folds (76%) -- confirms the sign
constraint is genuinely gone. Pooled: log loss 0.9726, Brier 0.5780,
accuracy 53.74% -- all slightly worse than the original model's 0.9678 /
0.5746 / 54.26%, i.e. removing the constraint the market itself doesn't
seem to need costs a small amount of fit quality elsewhere.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from sklearn.preprocessing import label_binarize

DATA_PATH = "data/processed/all_seasons_14window_ppg.csv"
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

_EPS = 1e-10


def _bivariate_normal_cdf(z1, z2, rho):
    """Bivariate standard normal CDF at correlation rho, vectorized over
    paired (z1, z2) arrays."""
    rho = float(np.clip(rho, -0.999, 0.999))
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    points = np.column_stack([z1, z2])
    return mvn.cdf(points)


def _copula_bivpois_logpmf(x, y, lam1, lam2, rho):
    """Log-pmf of two independent Poisson marginals joined by a Gaussian
    copula with correlation rho (the "rectangle" formula, see module
    docstring)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lam1 = np.asarray(lam1, dtype=float)
    lam2 = np.asarray(lam2, dtype=float)

    Fx = stats.poisson.cdf(x, lam1)
    Fx_1 = np.where(x > 0, stats.poisson.cdf(x - 1, lam1), 0.0)
    Fy = stats.poisson.cdf(y, lam2)
    Fy_1 = np.where(y > 0, stats.poisson.cdf(y - 1, lam2), 0.0)

    zx = stats.norm.ppf(np.clip(Fx, _EPS, 1 - _EPS))
    zx_1 = stats.norm.ppf(np.clip(Fx_1, _EPS, 1 - _EPS))
    zy = stats.norm.ppf(np.clip(Fy, _EPS, 1 - _EPS))
    zy_1 = stats.norm.ppf(np.clip(Fy_1, _EPS, 1 - _EPS))

    C_xy = _bivariate_normal_cdf(zx, zy, rho)
    C_x1y = _bivariate_normal_cdf(zx_1, zy, rho)
    C_xy1 = _bivariate_normal_cdf(zx, zy_1, rho)
    C_x1y1 = _bivariate_normal_cdf(zx_1, zy_1, rho)

    prob = C_xy - C_x1y - C_xy1 + C_x1y1
    prob = np.clip(prob, 1e-300, None)
    return np.log(prob)


class PoissonRegressionGoalsCopula:

    def __init__(self):
        self.k = len(HOME_COVARIATES)
        self.beta_home = None
        self.beta_away = None
        self.home_adv = None
        self.rho = None
        self.home_mean = self.home_std = self.away_mean = self.away_std = None

    def fit(self, matches_df):
        X_home_raw = matches_df[HOME_COVARIATES].astype(float)
        X_away_raw = matches_df[AWAY_COVARIATES].astype(float)

        self.home_mean = X_home_raw.mean(axis=0).values
        self.home_std = X_home_raw.std(axis=0).values
        self.away_mean = X_away_raw.mean(axis=0).values
        self.away_std = X_away_raw.std(axis=0).values
        self.home_std[self.home_std == 0] = 1
        self.away_std[self.away_std == 0] = 1

        X_home = X_home_raw.fillna(pd.Series(self.home_mean, index=HOME_COVARIATES)).values
        X_away = X_away_raw.fillna(pd.Series(self.away_mean, index=AWAY_COVARIATES)).values

        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        fthg = matches_df["FTHG"].values.astype(float)
        ftag = matches_df["FTAG"].values.astype(float)
        k = self.k

        def neg_log_likelihood(params):
            beta_home = params[:k]
            beta_away = params[k:2 * k]
            home_adv = params[2 * k]
            z = params[2 * k + 1]

            lam1 = np.exp(home_adv + X_home_std @ beta_home)
            lam2 = np.exp(X_away_std @ beta_away)
            rho = np.tanh(z)

            ll = _copula_bivpois_logpmf(fthg, ftag, lam1, lam2, rho)
            return -ll.sum()

        x0 = np.zeros(2 * k + 2)
        x0[2 * k] = 0.2
        x0[2 * k + 1] = 0.0

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

        self.beta_home = result.x[:k]
        self.beta_away = result.x[k:2 * k]
        self.home_adv = result.x[2 * k]
        self.rho = np.tanh(result.x[2 * k + 1])
        return self

    def predict_proba(self, matches_df, max_goals=10):
        """Returns an (n_matches, 3) array, columns ordered [H, D, A]."""
        X_home_raw = matches_df[HOME_COVARIATES].astype(float)
        X_away_raw = matches_df[AWAY_COVARIATES].astype(float)
        X_home = X_home_raw.fillna(pd.Series(self.home_mean, index=HOME_COVARIATES)).values
        X_away = X_away_raw.fillna(pd.Series(self.away_mean, index=AWAY_COVARIATES)).values
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        lam1_all = np.exp(self.home_adv + X_home_std @ self.beta_home)
        lam2_all = np.exp(X_away_std @ self.beta_away)

        xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
        x_flat, y_flat = xs.ravel().astype(float), ys.ravel().astype(float)

        probs = []
        for lam1, lam2 in zip(lam1_all, lam2_all):
            lam1_arr = np.full_like(x_flat, lam1)
            lam2_arr = np.full_like(x_flat, lam2)
            log_pmf = _copula_bivpois_logpmf(x_flat, y_flat, lam1_arr, lam2_arr, self.rho)
            grid = np.exp(log_pmf).reshape(max_goals + 1, max_goals + 1)
            grid = grid / grid.sum()

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
    fitted_rhos = []

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

            model = PoissonRegressionGoalsCopula().fit(train_df)
            fitted_rhos.append(model.rho)

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
                "rho": model.rho,
            })

        print(f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)}), "
              f"last rho={model.rho:.4f}")

    chunk_df = pd.DataFrame(chunk_rows)

    print(f"\n=== Within-season trend: mean model log loss by chunk index (0=start of season, {N_CHUNKS - 1}=end) ===")
    for c_idx in range(N_CHUNKS):
        vals = per_chunk_idx_ll[c_idx]
        if vals:
            print(f"chunk {c_idx}: mean_log_loss={np.mean(vals):.4f}  (n_seasons={len(vals)})")

    fitted_rhos = np.array(fitted_rhos)
    print(f"\n=== Fitted rho across all {len(fitted_rhos)} folds ===")
    print(f"range: [{fitted_rhos.min():.4f}, {fitted_rhos.max():.4f}], mean: {fitted_rhos.mean():.4f}")
    print(f"negative on {(fitted_rhos < 0).sum()}/{len(fitted_rhos)} folds "
          f"({(fitted_rhos < 0).mean():.1%})")

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

    print(
        "\n(Compare against Poisson_Covariates_Bivariate.py's reported result: "
        "0.9678 pooled log loss / 0.5746 Brier / 54.26% accuracy, "
        "and Poisson_Bivariate_MeanImpute.py's isolated-imputation-only result.)"
    )

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
