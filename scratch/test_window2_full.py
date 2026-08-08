"""
Poisson_Covariates.py, but replacing the Dixon-Coles low-score "tau" patch
with a genuine bivariate Poisson (Karlis & Ntzoufras 2003 trivariate
reduction), the approach used in Koopman & Lit (2015), "A Dynamic Bivariate
Poisson Model for Analysing and Forecasting Match Results in the EPL".

Dixon-Coles models home/away goals as INDEPENDENT Poisson counts, then
patches in a hand-picked correction (rho) that only touches the 4
lowest-scoring cells (0-0, 1-0, 0-1, 1-1) of the scoreline grid. The
bivariate Poisson instead builds correlation into the generative process
itself: three independent Poisson variables X1, X2, X3 with

    Home = X1 + X3
    Away = X2 + X3

so X3 is a goals component shared by both teams (e.g. game pace/tempo),
inducing Cov(Home, Away) = lambda3 across EVERY scoreline, not just the
low-scoring corner. log(lambda1) and log(lambda2) are driven by the same
covariates as Poisson_Covariates.py (Elo, xG, table position); lambda3 is
fit as a single shared constant (theta), replacing Dixon-Coles' rho as the
model's one correlation parameter -- same parameter count as before.

CAVEAT going in (known limitation of this construction, not a bug): this
trivariate-reduction form can only produce lambda3 >= 0, i.e. it can only
add NON-NEGATIVE correlation between home and away goals, whereas Dixon-
Coles' fitted rho on this data has consistently come out negative (see
Poisson.py / Poisson_Covariates.py, rho = -0.0004 / -0.009). So in theory
this model can't replicate that specific low-score correction.

RESULT (WINDOW=3, vs. Bet365): in practice that caveat didn't matter --
this is a clear, across-the-board improvement over the Dixon-Coles-tau
version. Pooled log loss dropped from 0.9769 to 0.9719 (fitted lambda3 =
0.1914, comfortably positive), Brier improved 0.5808 -> 0.5776, and
accuracy improved 53.68% -> 54.05%, all closer to Bet365. The one metric
that got WORSE is single-match win-rate against the market: 49.66% ->
47.22%. That's the win-rate/log-loss divergence this project has run into
before -- log loss is the authoritative metric here (it's what's actually
being optimized, and it's magnitude-sensitive in a way win-rate isn't), so
this is a real improvement, not a wash. New best model in this project.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from sklearn.preprocessing import label_binarize
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_with_bookies.csv"
WINDOW = 2

HOME_COVARIATES = ["Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_TablePosDiff"]
AWAY_COVARIATES = ["Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_TablePosDiff"]


def _bivpois_logpmf(x, y, lam1, lam2, lam3):
    """Log-pmf of the bivariate Poisson (trivariate reduction) at integer
    arrays x, y, given per-observation lam1/lam2 and scalar (or
    broadcastable) lam3. Vectorized over the inner sum-over-i by looping
    only over i = 0..min(x, y).max(), not over individual observations."""
    min_xy = np.minimum(x, y)
    max_k = int(min_xy.max())

    log_terms = np.full((max_k + 1, len(x)), -np.inf)
    log_ratio = np.log(lam3) - np.log(lam1) - np.log(lam2)
    for i in range(max_k + 1):
        mask = min_xy >= i
        xi, yi = x[mask], y[mask]
        logC_x = gammaln(xi + 1) - gammaln(i + 1) - gammaln(xi - i + 1)
        logC_y = gammaln(yi + 1) - gammaln(i + 1) - gammaln(yi - i + 1)
        ratio_term = i * (log_ratio if np.isscalar(log_ratio) else log_ratio[mask])
        log_terms[i, mask] = logC_x + logC_y + gammaln(i + 1) + ratio_term

    logS = logsumexp(log_terms, axis=0)
    return -(lam1 + lam2 + lam3) + x * np.log(lam1) - gammaln(x + 1) + y * np.log(lam2) - gammaln(y + 1) + logS


class PoissonRegressionGoalsBivariate:

    def __init__(self):
        self.k = len(HOME_COVARIATES)
        self.beta_home = None
        self.beta_away = None
        self.home_adv = None
        self.theta = None  # lambda3 = exp(theta), the shared/covariance component
        self.home_mean = self.home_std = self.away_mean = self.away_std = None

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
            theta = params[2 * k + 1]

            lam1 = np.exp(home_adv + X_home_std @ beta_home)
            lam2 = np.exp(X_away_std @ beta_away)
            lam3 = np.exp(theta)

            ll = _bivpois_logpmf(fthg, ftag, lam1, lam2, lam3)
            return -ll.sum()

        x0 = np.zeros(2 * k + 2)
        x0[2 * k] = 0.2       # home advantage init
        x0[2 * k + 1] = -3.0  # small initial lambda3 (~0.05)

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

        self.beta_home = result.x[:k]
        self.beta_away = result.x[k:2 * k]
        self.home_adv = result.x[2 * k]
        self.theta = result.x[2 * k + 1]
        return self

    def predict_proba(self, matches_df, max_goals=10):
        """Returns an (n_matches, 3) array, columns ordered [H, D, A]."""
        X_home = matches_df[HOME_COVARIATES].fillna(0).values
        X_away = matches_df[AWAY_COVARIATES].fillna(0).values
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        lam1_all = np.exp(self.home_adv + X_home_std @ self.beta_home)
        lam2_all = np.exp(X_away_std @ self.beta_away)
        lam3 = np.exp(self.theta)

        xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
        x_flat, y_flat = xs.ravel().astype(float), ys.ravel().astype(float)

        probs = []
        for lam1, lam2 in zip(lam1_all, lam2_all):
            lam1_arr = np.full_like(x_flat, lam1)
            lam2_arr = np.full_like(x_flat, lam2)
            log_pmf = _bivpois_logpmf(x_flat, y_flat, lam1_arr, lam2_arr, lam3)
            grid = np.exp(log_pmf).reshape(max_goals + 1, max_goals + 1)
            grid = grid / grid.sum()  # renormalize for the max_goals truncation

            p_home = np.tril(grid, -1).sum()
            p_draw = np.trace(grid)
            p_away = np.triu(grid, 1).sum()
            probs.append((p_home, p_draw, p_away))
        return np.array(probs)


def single_run_sanity_check(all_df):
    train_seasons = ["22-23", "23-24", "24-25"]
    train_df = all_df[all_df["Season"].isin(train_seasons)]

    model = PoissonRegressionGoalsBivariate().fit(train_df)

    print(f"Home advantage (log-space): {model.home_adv:.4f}  (~{np.exp(model.home_adv):.2f}x home scoring)")
    print(f"Lambda3 (shared/covariance component): {np.exp(model.theta):.4f}  (theta={model.theta:.4f})")
    print(f"\nHome-goals equation coefficients {HOME_COVARIATES}:")
    print(model.beta_home)
    print(f"\nAway-goals equation coefficients {AWAY_COVARIATES}:")
    print(model.beta_away)


def walk_forward_vs_bet365(all_df):
    seasons = sorted(all_df["Season"].unique())

    fold_rows = []
    all_model_ll, all_b365_ll = [], []
    all_model_brier, all_b365_brier = [], []
    all_model_correct, all_b365_correct = [], []

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]

        train_df = all_df[all_df["Season"].isin(train_seasons)]
        test_df = all_df[all_df["Season"] == test_season]

        model = PoissonRegressionGoalsBivariate().fit(train_df)

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
        model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
        b365_brier = ((fair_market_proba - y_onehot) ** 2).sum(axis=1)
        model_pred = np.array(classes_order)[proba.argmax(axis=1)]
        b365_pred = np.array(classes_order)[fair_market_proba.argmax(axis=1)]
        model_correct = (model_pred == y_true)
        b365_correct = (b365_pred == y_true)

        all_model_ll.append(model_ll)
        all_b365_ll.append(b365_ll)
        all_model_brier.append(model_brier)
        all_b365_brier.append(b365_brier)
        all_model_correct.append(model_correct)
        all_b365_correct.append(b365_correct)

        fold_rows.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            "n_test": len(test_df),
            "model_log_loss": model_ll.mean(),
            "bet365_log_loss": b365_ll.mean(),
            "model_acc": model_correct.mean(),
            "bet365_acc": b365_correct.mean(),
        })

        print(
            f"{test_season} (train {train_seasons[0]}..{train_seasons[-1]}): "
            f"model={model_ll.mean():.4f} bet365={b365_ll.mean():.4f} "
            f"| acc model={model_correct.mean():.2%} bet365={b365_correct.mean():.2%}"
        )

    fold_df = pd.DataFrame(fold_rows)
    print("\n=== Per-fold summary ===")
    print(fold_df[["test_season", "n_test", "model_log_loss", "bet365_log_loss", "model_acc", "bet365_acc"]].round(4))

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    model_brier_all = np.concatenate(all_model_brier)
    b365_brier_all = np.concatenate(all_b365_brier)
    model_correct_all = np.concatenate(all_model_correct)
    b365_correct_all = np.concatenate(all_b365_correct)

    print(f"\n=== Pooled across all {len(fold_df)} folds ({len(model_ll_all)} test matches) ===")
    print(f"Model log loss:  {model_ll_all.mean():.4f}")
    print(f"Bet365 log loss: {b365_ll_all.mean():.4f}")
    print(f"Model Brier:     {model_brier_all.mean():.4f}")
    print(f"Bet365 Brier:    {b365_brier_all.mean():.4f}")
    print(f"Model accuracy (1X2):  {model_correct_all.mean():.2%}")
    print(f"Bet365 accuracy (1X2): {b365_correct_all.mean():.2%}")

    t_stat, p_value = stats.ttest_rel(model_ll_all, b365_ll_all)
    print(f"\nPaired t-test (log loss, model vs Bet365): t={t_stat:.3f}, p={p_value:.6f}")

    diff = model_ll_all - b365_ll_all
    rng = np.random.default_rng(42)
    n = len(diff)
    boot_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI for mean log-loss diff (model - Bet365): [{ci_low:.4f}, {ci_high:.4f}]")

    model_wins = (model_ll_all < b365_ll_all).sum()
    b365_wins = (b365_ll_all < model_ll_all).sum()
    print(f"\nModel better on {model_wins}/{n} matches, Bet365 better on {b365_wins}/{n} matches")
    print(f"Model better on {model_wins/n:.2%} of matches, Bet365 better on {b365_wins/n:.2%} of matches")
    print(
        "\n(Compare against Poisson_Covariates.py's Dixon-Coles-tau version: "
        "0.9769 pooled log loss / 49.66% win-rate / 53.68% accuracy.)"
    )


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("=" * 70)
    print("SINGLE RUN SANITY CHECK")
    print("=" * 70)
    single_run_sanity_check(all_df)

    print("\n" + "=" * 70)
    print("WALK-FORWARD VS. BET365")
    print("=" * 70)
    walk_forward_vs_bet365(all_df)


if __name__ == "__main__":
    main()
