import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.preprocessing import label_binarize
from scipy import stats

HOME_COVARIATES = ["Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_TablePosDiff"]
AWAY_COVARIATES = ["Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_TablePosDiff"]
import sys
XI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0025


def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    elif x == 0 and y == 1: return 1 + lam * rho
    elif x == 1 and y == 0: return 1 + mu * rho
    elif x == 1 and y == 1: return 1 - rho
    return 1.0


def fit_weighted(matches_df, xi=XI):
    X_home = matches_df[HOME_COVARIATES].fillna(0).values
    X_away = matches_df[AWAY_COVARIATES].fillna(0).values
    home_mean, home_std = X_home.mean(axis=0), X_home.std(axis=0)
    away_mean, away_std = X_away.mean(axis=0), X_away.std(axis=0)
    home_std[home_std == 0] = 1
    away_std[away_std == 0] = 1
    X_home_std = (X_home - home_mean) / home_std
    X_away_std = (X_away - away_mean) / away_std

    fthg = matches_df["FTHG"].values
    ftag = matches_df["FTAG"].values
    k = len(HOME_COVARIATES)

    ref_date = matches_df["Date"].max()
    days_ago = (ref_date - matches_df["Date"]).dt.days.values
    weights = np.exp(-xi * days_ago)

    def neg_log_likelihood(params):
        beta_home = params[:k]
        beta_away = params[k:2*k]
        home_adv = params[2*k]
        rho = params[2*k+1]

        lam = np.exp(home_adv + X_home_std @ beta_home)
        mu = np.exp(X_away_std @ beta_away)

        ll = poisson.logpmf(fthg, lam) + poisson.logpmf(ftag, mu)
        tau_vals = np.array([tau(x,y,l,m,rho) for x,y,l,m in zip(fthg,ftag,lam,mu)])
        tau_vals = np.clip(tau_vals, 1e-10, None)
        ll += np.log(tau_vals)
        return -(weights * ll).sum()

    x0 = np.zeros(2*k+2)
    x0[2*k] = 0.2
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
    print('converged:', result.success)
    return result.x, home_mean, home_std, away_mean, away_std


def predict_proba(params, home_mean, home_std, away_mean, away_std, matches_df, max_goals=10):
    k = len(HOME_COVARIATES)
    beta_home = params[:k]
    beta_away = params[k:2*k]
    home_adv = params[2*k]
    rho = params[2*k+1]

    X_home = matches_df[HOME_COVARIATES].fillna(0).values
    X_away = matches_df[AWAY_COVARIATES].fillna(0).values
    X_home_std = (X_home - home_mean) / home_std
    X_away_std = (X_away - away_mean) / away_std

    lam_all = np.exp(home_adv + X_home_std @ beta_home)
    mu_all = np.exp(X_away_std @ beta_away)

    probs = []
    for lam, mu in zip(lam_all, mu_all):
        home_pmf = poisson.pmf(np.arange(max_goals+1), lam)
        away_pmf = poisson.pmf(np.arange(max_goals+1), mu)
        grid = np.outer(home_pmf, away_pmf)
        for x in range(2):
            for y in range(2):
                grid[x,y] *= tau(x,y,lam,mu,rho)
        grid = grid / grid.sum()
        probs.append((np.tril(grid,-1).sum(), np.trace(grid), np.triu(grid,1).sum()))
    return np.array(probs)


WINDOW = 3
all_df = pd.read_csv("dataset/all_seasons_with_bookies.csv", parse_dates=["Date"])
seasons = sorted(all_df["Season"].unique())

all_model_ll, all_b365_ll = [], []
for i in range(WINDOW, len(seasons)):
    train_seasons = seasons[i-WINDOW:i]
    test_season = seasons[i]
    train_df = all_df[all_df["Season"].isin(train_seasons)]
    test_df = all_df[all_df["Season"] == test_season]

    params, hm, hs, am, as_ = fit_weighted(train_df)
    proba = predict_proba(params, hm, hs, am, as_, test_df)
    classes_order = ["H","D","A"]

    overround = 1/test_df["B365HomeOdds"] + 1/test_df["B365DrawOdds"] + 1/test_df["B365AwayOdds"]
    fair_h = (1/test_df["B365HomeOdds"])/overround
    fair_d = (1/test_df["B365DrawOdds"])/overround
    fair_a = (1/test_df["B365AwayOdds"])/overround
    fair_market_proba = np.column_stack([fair_h.values, fair_d.values, fair_a.values])

    y_true = test_df["FTR"].values
    y_onehot = label_binarize(y_true, classes=classes_order)
    eps = 1e-15
    model_ll = -np.log(np.clip((proba*y_onehot).sum(axis=1), eps, 1))
    b365_ll = -np.log(np.clip((fair_market_proba*y_onehot).sum(axis=1), eps, 1))
    all_model_ll.append(model_ll)
    all_b365_ll.append(b365_ll)
    print(f"{test_season}: model={model_ll.mean():.4f} bet365={b365_ll.mean():.4f}")

model_ll_all = np.concatenate(all_model_ll)
b365_ll_all = np.concatenate(all_b365_ll)
print(f"\nPooled model: {model_ll_all.mean():.4f}  Pooled bet365: {b365_ll_all.mean():.4f}")
t_stat, p_value = stats.ttest_rel(model_ll_all, b365_ll_all)
print(f"p={p_value:.6f}")
model_wins = (model_ll_all < b365_ll_all).sum()
n = len(model_ll_all)
print(f"Model better on {model_wins}/{n} ({model_wins/n:.2%})")
