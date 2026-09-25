"""
Round-to-round "odds maker": refreshes current-season data, retrains the
model when a new evaluation chunk begins (every ~7-8 rounds, matching the
validated WINDOW=3/N_CHUNKS=5 walk-forward protocol used everywhere else
in this project), predicts the next unplayed round's fixtures, and
appends the results (H/D/A probabilities + fair odds + a full scoreline
fair-odds grid) to a persistent CSV log for later evaluation once actual
results are known.

Usage: just run this file (from the project root: `python live/predict_round.py`).
Safe to run every week -- it always recomputes "the next unplayed round"
fresh from the fixture feed, so it doesn't matter if you skip a week or
run it twice in one day (it'll just predict the same round again with
whatever's newly available, and append another logged row).
"""

import json
import os
import sys
from datetime import datetime, timezone

# Windows' default console codepage can't encode some characters emitted by
# third-party retry/logging paths (e.g. soccerdata) -- make stdout/stderr
# tolerant instead of letting an unrelated log line crash the whole run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "models", "maxxing"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "pipeline"))
from Poisson_Bivariate_MeanImpute import (  # noqa: E402
    PoissonRegressionGoalsMeanImpute, _bivpois_logpmf, HOME_COVARIATES, AWAY_COVARIATES,
)

from build_full_dataset_no_loss import SEASONS as HISTORICAL_SEASONS  # noqa: E402
from live.fetch_fixtures import fetch_season_fixtures, next_unplayed_round  # noqa: E402
from live.fetch_current_season_matches import fetch_current_season_matches  # noqa: E402
from live.build_live_features import build_training_and_prediction_frames  # noqa: E402
from live.build_analysis_workbook import main as rebuild_analysis_workbook, OUTPUT_XLSX  # noqa: E402

CURRENT_SEASON = "26-27"
FIXTURE_SLUG = "epl-2026"
TRAIN_SEASONS = HISTORICAL_SEASONS[-3:]  # 3 most recent complete seasons -- fixed all season, same as WINDOW=3

TOTAL_ROUNDS = 38
N_CHUNKS = 5
ROUND_CHUNKS = np.array_split(np.arange(1, TOTAL_ROUNDS + 1), N_CHUNKS)

STATE_DIR = "live/state"
STATE_JSON = f"{STATE_DIR}/model_state.json"
STATE_NPZ = f"{STATE_DIR}/model_params.npz"
LOG_CSV = "live/predictions_log.csv"
READABLE_LOG_CSV = "live/predictions_readable.csv"

MAX_GOALS_GRID = 6


def chunk_index_for_round(round_number):
    for i, chunk in enumerate(ROUND_CHUNKS):
        if round_number in chunk:
            return i
    raise ValueError(f"round {round_number} outside 1..{TOTAL_ROUNDS}")


def load_state():
    if not os.path.exists(STATE_JSON):
        return None
    with open(STATE_JSON) as f:
        return json.load(f)


def save_state(chunk_index, rounds_played):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_JSON, "w") as f:
        json.dump({
            "season": CURRENT_SEASON,
            "chunk_index": chunk_index,
            "train_seasons": TRAIN_SEASONS,
            "rounds_played": rounds_played,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def load_model():
    npz = np.load(STATE_NPZ)
    model = PoissonRegressionGoalsMeanImpute()
    model.beta_home = npz["beta_home"]
    model.beta_away = npz["beta_away"]
    model.home_adv = float(npz["home_adv"])
    model.theta = float(npz["theta"])
    model.home_mean = npz["home_mean"]
    model.home_std = npz["home_std"]
    model.away_mean = npz["away_mean"]
    model.away_std = npz["away_std"]
    return model


def save_model(model):
    os.makedirs(STATE_DIR, exist_ok=True)
    np.savez(
        STATE_NPZ,
        beta_home=model.beta_home, beta_away=model.beta_away,
        home_adv=model.home_adv, theta=model.theta,
        home_mean=model.home_mean, home_std=model.home_std,
        away_mean=model.away_mean, away_std=model.away_std,
    )


def scoreline_grids(model, predict_df, max_goals=MAX_GOALS_GRID):
    X_home = predict_df[HOME_COVARIATES].astype(float).fillna(pd.Series(model.home_mean, index=HOME_COVARIATES)).values
    X_away = predict_df[AWAY_COVARIATES].astype(float).fillna(pd.Series(model.away_mean, index=AWAY_COVARIATES)).values
    X_home_std = (X_home - model.home_mean) / model.home_std
    X_away_std = (X_away - model.away_mean) / model.away_std
    lam1_all = np.exp(model.home_adv + X_home_std @ model.beta_home)
    lam2_all = np.exp(X_away_std @ model.beta_away)
    lam3 = np.exp(model.theta)

    xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
    x_flat, y_flat = xs.ravel().astype(float), ys.ravel().astype(float)

    grids = []
    for lam1, lam2 in zip(lam1_all, lam2_all):
        log_pmf = _bivpois_logpmf(x_flat, y_flat, np.full_like(x_flat, lam1), np.full_like(x_flat, lam2), lam3)
        grid = np.exp(log_pmf).reshape(max_goals + 1, max_goals + 1)
        grid = grid / grid.sum()
        grids.append(grid)
    return grids


def grid_to_fair_odds_dict(grid):
    """{'H-A goals': fair_odds}, e.g. '2-1': 8.33"""
    out = {}
    for h in range(grid.shape[0]):
        for a in range(grid.shape[1]):
            p = grid[h, a]
            out[f"{h}-{a}"] = round(1.0 / p, 2) if p > 1e-12 else None
    return out


def main():
    print(f"=== Live prediction run for {CURRENT_SEASON}, {datetime.now().isoformat()} ===")

    fixtures = fetch_season_fixtures(FIXTURE_SLUG)
    round_to_predict = next_unplayed_round(fixtures)
    if round_to_predict is None:
        print("Season complete -- no unplayed rounds left.")
        return
    print(f"Next round to predict: {round_to_predict}")

    chunk_idx = chunk_index_for_round(round_to_predict)
    print(f"Round {round_to_predict} falls in chunk {chunk_idx} (of {N_CHUNKS})")

    state = load_state()
    need_retrain = (
        state is None
        or state.get("season") != CURRENT_SEASON
        or state.get("chunk_index") != chunk_idx
    )

    current_played = fetch_current_season_matches(CURRENT_SEASON)
    # A round can be partially played already (rearranged fixtures, one
    # match moved earlier) -- only predict what's genuinely still upcoming.
    round_fixtures = fixtures[fixtures["RoundNumber"] == round_to_predict]
    already_played = round_fixtures[round_fixtures["Played"]]
    if not already_played.empty:
        print(f"Skipping {len(already_played)} already-played fixture(s) in round {round_to_predict}: "
              + ", ".join(f"{r.HomeTeam} vs {r.AwayTeam}" for r in already_played.itertuples()))
    upcoming = round_fixtures[~round_fixtures["Played"]][["Date", "HomeTeam", "AwayTeam"]].copy()

    train_df, predict_df = build_training_and_prediction_frames(
        current_season=CURRENT_SEASON,
        current_season_played_df=current_played,
        upcoming_fixtures_df=upcoming,
        train_seasons=TRAIN_SEASONS,
    )

    rounds_played = int(fixtures.loc[fixtures["Played"], "RoundNumber"].nunique())

    if need_retrain:
        print(f"Retraining (new chunk boundary reached) on {len(train_df)} matches "
              f"({TRAIN_SEASONS} + {CURRENT_SEASON} played-so-far: {rounds_played} rounds)...")
        model = PoissonRegressionGoalsMeanImpute().fit(train_df)
        save_model(model)
        save_state(chunk_idx, rounds_played)
        print(f"Retrained. home_adv={model.home_adv:.4f}, theta={model.theta:.4f}")
    else:
        print(f"Reusing model cached for chunk {chunk_idx} (state trained_at={state['trained_at']})")
        model = load_model()

    proba = model.predict_proba(predict_df)
    grids = scoreline_grids(model, predict_df)

    run_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, (_, fixture) in enumerate(predict_df.reset_index(drop=True).iterrows()):
        p_h, p_d, p_a = proba[i]
        rows.append({
            "run_timestamp": run_ts,
            "season": CURRENT_SEASON,
            "round": round_to_predict,
            "chunk_index": chunk_idx,
            "retrained_this_run": need_retrain,
            "match_date": fixture["Date"],
            "home_team": fixture["HomeTeam"],
            "away_team": fixture["AwayTeam"],
            "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
            "fair_odds_home": round(1 / p_h, 2) if p_h > 1e-12 else None,
            "fair_odds_draw": round(1 / p_d, 2) if p_d > 1e-12 else None,
            "fair_odds_away": round(1 / p_a, 2) if p_a > 1e-12 else None,
            "scoreline_fair_odds_json": json.dumps(grid_to_fair_odds_dict(grids[i])),
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_result": None,
        })

    log_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    write_header = not os.path.exists(LOG_CSV)
    log_df.to_csv(LOG_CSV, mode="a", header=write_header, index=False)
    print(f"\nAppended {len(log_df)} predictions to {LOG_CSV}")

    # Plain, human-readable companion log -- just match + odds, no JSON
    # blob, so it opens cleanly in Excel/a text editor.
    readable_df = log_df[[
        "round", "match_date", "home_team", "away_team",
        "fair_odds_home", "fair_odds_draw", "fair_odds_away",
        "actual_home_goals", "actual_away_goals", "actual_result",
    ]].rename(columns={
        "fair_odds_home": "odds_home", "fair_odds_draw": "odds_draw", "fair_odds_away": "odds_away",
    })
    write_header_readable = not os.path.exists(READABLE_LOG_CSV)
    readable_df.to_csv(READABLE_LOG_CSV, mode="a", header=write_header_readable, index=False)
    print(f"Appended {len(readable_df)} rows to {READABLE_LOG_CSV}")

    try:
        rebuild_analysis_workbook()
    except PermissionError:
        print(f"\nCould not rebuild {OUTPUT_XLSX} -- it's probably still open in Excel. "
              f"Close it and re-run live/build_analysis_workbook.py to refresh it.")

    print(f"\n=== Round {round_to_predict} predictions ===")
    for _, r in log_df.iterrows():
        print(f"{r['home_team']:>16} vs {r['away_team']:<16}  "
              f"H:{r['p_home']:.1%} ({r['fair_odds_home']})  "
              f"D:{r['p_draw']:.1%} ({r['fair_odds_draw']})  "
              f"A:{r['p_away']:.1%} ({r['fair_odds_away']})")


if __name__ == "__main__":
    main()
