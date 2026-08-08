"""
Same intra-season cumulative sliding walk-forward as
Poisson_Bivariate_intraseason_walkforward.py (WINDOW=3, N_CHUNKS=5, PPG-based
covariates), but compared against Bet365, Bet&Win, and Pinnacle INDIVIDUALLY
instead of Bet365 + the market average -- does the model beat any single
bookmaker, not just the pooled market?

Only these three: they're the only bookmakers with odds in every season this
script tests on (17-18..25-26) -- football-data.co.uk's tracked bookmaker
list changes over time (e.g. William Hill/VC Bet/Interwetten drop out in the
most recent 1-2 seasons), so this is the largest common, apples-to-apples set.
BW/PS odds aren't in the working dataset (only Bet365 and the market average
are), so they're merged in here from the raw per-season CSVs.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from scipy import stats

from Poisson_Covariates_Bivariate import PoissonRegressionGoalsBivariate

DATA_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\dataset\all_seasons_14window_ppg.csv"
WINDOW = 3
N_CHUNKS = 5
BOOKIES = ["B365", "BW", "PS"]  # Bet365, Bet&Win, Pinnacle -- only ones with full coverage 17-18..25-26


def load_bookie_odds(seasons):
    frames = []
    for season in seasons:
        cols = ["Date", "HomeTeam", "AwayTeam"] + [f"{b}{r}" for b in BOOKIES for r in "HDA"]
        df = pd.read_csv(f"pl{season}.csv", usecols=cols)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    seasons = sorted(all_df["Season"].unique())

    odds = load_bookie_odds(seasons)
    all_df = all_df.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    classes_order = ["H", "D", "A"]
    all_model_ll = []
    # Paired per-bookie (model_ll, bookie_ll), NaN-filtered per bookie -- some
    # bookmakers are missing odds for a chunk of matches in a given season
    # (e.g. BW: 141/380 missing in 24-25; PS: 170/380 missing in 25-26, still
    # in progress), so each bookie's comparison uses only its own valid rows.
    paired_model_ll = {b: [] for b in BOOKIES}
    paired_bookie_ll = {b: [] for b in BOOKIES}

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

            model = PoissonRegressionGoalsBivariate().fit(train_df)
            proba = model.predict_proba(test_chunk)

            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=classes_order)
            eps = 1e-15
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            all_model_ll.append(model_ll)

            for b in BOOKIES:
                valid = test_chunk[[f"{b}H", f"{b}D", f"{b}A"]].notna().all(axis=1).values
                if not valid.any():
                    continue
                sub = test_chunk[valid]
                sub_onehot = y_onehot[valid]

                overround = 1 / sub[f"{b}H"] + 1 / sub[f"{b}D"] + 1 / sub[f"{b}A"]
                fair = np.column_stack([
                    ((1 / sub[f"{b}H"]) / overround).values,
                    ((1 / sub[f"{b}D"]) / overround).values,
                    ((1 / sub[f"{b}A"]) / overround).values,
                ])
                bookie_ll = -np.log(np.clip((fair * sub_onehot).sum(axis=1), eps, 1))
                paired_bookie_ll[b].append(bookie_ll)
                paired_model_ll[b].append(model_ll[valid])

    model_ll_all = np.concatenate(all_model_ll)
    print(f"Model overall log_loss={model_ll_all.mean():.4f} (n={len(model_ll_all)})\n")

    for b in BOOKIES:
        m = np.concatenate(paired_model_ll[b])
        k = np.concatenate(paired_bookie_ll[b])
        t_stat, p_value = stats.ttest_rel(m, k)
        wins = (m < k).sum()
        n = len(m)
        print(
            f"{b:5s} n={n:4d}  model={m.mean():.4f} vs {b}={k.mean():.4f}  "
            f"t={t_stat:+.3f} p={p_value:.4f}  model_wins={wins}/{n} ({wins/n:.1%})"
        )


if __name__ == "__main__":
    main()
