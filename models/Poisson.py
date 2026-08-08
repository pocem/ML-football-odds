"""
Dixon-Coles (1997) Poisson goal model.

Every other model in this project is discriminative: learn P(H/D/A | features)
directly as a classifier. This is generative instead -- it models the actual
goal-scoring process football is built from. Each team gets an attack strength
and a defense weakness; home/away goals are modeled as two Poisson-distributed
counts driven by those ratings plus a home-advantage term; a small correlation
correction (rho) fixes the fact that pure independent Poisson underestimates
low-scoring draws like 0-0/1-1. Summing the resulting scoreline grid gives
P(H)/P(D)/P(A) -- but also gives full correct-score/over-under probabilities
for free, which a classifier never does.

Note this model only ever consumes HomeTeam/AwayTeam/FTHG/FTAG -- it doesn't
take any of the engineered feature columns (Elo, rolling xG, PPG, etc.), it
fits its own per-team attack/defense ratings straight from goals scored. The
covariate-driven version lives in Poisson_Covariates_Bivariate.py instead.

Now on DATA_PATH = all_seasons_14window_ppg.csv (the project's current
canonical dataset) and the "classic" intra-season cumulative sliding
walk-forward (same ablation as RF_intraseason_walkforward.py /
Poisson_Bivariate_intraseason_walkforward.py / FFNN.py / LogisticRegression.py
/ RF.py / XGBoost.py / SVM_UMAP.py): each test season is split into N_CHUNKS
chronological pieces, and training grows every chunk (prior WINDOW seasons +
every earlier chunk of the current season), instead of jumping straight from
"last season" to predicting the whole next season in one shot. No separate
validation-season carve-out here, same as the original season-level version --
Dixon-Coles has ~2*n_teams+2 parameters fit by MLE on 1000+ matches, nowhere
near tree-ensemble capacity, so the train/test overfitting gap that motivated
a validation fold for RF/XGBoost isn't expected to be a concern here either.

RESULT: filled in after running -- see bottom of this docstring.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.preprocessing import label_binarize
from scipy import stats

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3      # prior full seasons used as the historical baseline
N_CHUNKS = 5    # chronological chunks the current test season is split into


class DixonColes:

    def __init__(self):
        self.teams = None
        self.team_idx = None
        self.attack = None
        self.defense = None
        self.home_adv = None
        self.rho = None

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
        self.teams = sorted(set(matches_df["HomeTeam"]).union(matches_df["AwayTeam"]))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = matches_df["HomeTeam"].map(self.team_idx).values
        away_idx = matches_df["AwayTeam"].map(self.team_idx).values
        fthg = matches_df["FTHG"].values
        ftag = matches_df["FTAG"].values

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            # log(lambda) = home_adv + attack_home + defense_away (higher defense_i = weaker defense)
            lam = np.exp(home_adv + attack[home_idx] + defense[away_idx])
            mu = np.exp(attack[away_idx] + defense[home_idx])

            ll = poisson.logpmf(fthg, lam) + poisson.logpmf(ftag, mu)
            tau_vals = np.array([
                self._tau(x, y, l, m, rho) for x, y, l, m in zip(fthg, ftag, lam, mu)
            ])
            tau_vals = np.clip(tau_vals, 1e-10, None)
            ll += np.log(tau_vals)
            return -ll.sum()

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.2  # home advantage init

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

        self.attack = result.x[:n]
        self.defense = result.x[n:2 * n]
        self.home_adv = result.x[2 * n]
        self.rho = result.x[2 * n + 1]
        return self

    def _team_params(self, team):
        # Unseen team (e.g. newly promoted with no history in the training
        # window) falls back to average strength -- attack=0, defense=0.
        idx = self.team_idx.get(team)
        if idx is None:
            return 0.0, 0.0
        return self.attack[idx], self.defense[idx]

    def predict_proba_one(self, home_team, away_team, max_goals=10):
        atk_h, def_h = self._team_params(home_team)
        atk_a, def_a = self._team_params(away_team)

        lam = np.exp(self.home_adv + atk_h + def_a)
        mu = np.exp(atk_a + def_h)

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
        return p_home, p_draw, p_away

    def predict_proba(self, matches_df):
        """Returns an (n_matches, 3) array, columns ordered [H, D, A]."""
        probs = [
            self.predict_proba_one(h, a)
            for h, a in zip(matches_df["HomeTeam"], matches_df["AwayTeam"])
        ]
        return np.array(probs)


def single_run_sanity_check(all_df):
    """Fit on 3 seasons, inspect the fitted team ratings, and hand-check
    one lopsided and one close matchup."""
    train_seasons = ["22-23", "23-24", "24-25"]
    train_df = all_df[all_df["Season"].isin(train_seasons)]

    dc = DixonColes().fit(train_df)

    print(f"Home advantage (log-space): {dc.home_adv:.4f}  (~{np.exp(dc.home_adv):.2f}x home scoring)")
    print(f"Rho (low-score correlation): {dc.rho:.6f}")

    ratings = pd.DataFrame({"team": dc.teams, "attack": dc.attack, "defense": dc.defense})
    ratings = ratings.sort_values("attack", ascending=False)
    print("\nTop 8 by attack strength:")
    print(ratings.head(8).to_string(index=False))
    print("\nBottom 8 by attack strength:")
    print(ratings.tail(8).to_string(index=False))

    print("\n--- Sanity checks ---")
    p_h, p_d, p_a = dc.predict_proba_one("Man City", "Southampton")
    print(f"Man City (H) vs Southampton (A): P(H)={p_h:.3f} P(D)={p_d:.3f} P(A)={p_a:.3f}")

    p_h, p_d, p_a = dc.predict_proba_one("Tottenham", "Brighton")
    print(f"Tottenham (H) vs Brighton (A):   P(H)={p_h:.3f} P(D)={p_d:.3f} P(A)={p_a:.3f}")


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
    """Intra-season cumulative sliding walk-forward: each test season is split
    into N_CHUNKS chronological pieces, and training grows every chunk (prior
    WINDOW full seasons + every earlier chunk of the current season), compared
    against de-vigged Bet365 odds AND de-vigged average-bookmaker odds."""
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

            dc = DixonColes().fit(train_df)

            proba = dc.predict_proba(test_chunk)  # columns: [H, D, A]
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

        print(
            f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)})"
        )

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

    print("=" * 70)
    print("SINGLE RUN SANITY CHECK")
    print("=" * 70)
    single_run_sanity_check(all_df)

    print("\n" + "=" * 70)
    print("WALK-FORWARD VS. MARKET (BET365 + AVG-BOOKIE)")
    print("=" * 70)
    walk_forward_vs_market(all_df)


if __name__ == "__main__":
    main()
