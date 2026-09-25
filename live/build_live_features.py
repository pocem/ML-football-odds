"""
Live feature builder for round-to-round prediction. Mirrors
build_full_dataset_no_loss.py's logic (imports its constants/helpers
directly rather than re-deriving them) but generalized to append the
CURRENT, in-progress season on top of the 12 completed historical
seasons, plus placeholder rows for whichever round is about to be
predicted (not yet played).

Because every rolling feature this model uses is a continuous,
never-reset-at-season-boundary EWMA, a placeholder row for an unplayed
fixture naturally inherits each team's latest known feature values just
by being appended at the end of that team's history, with its own
(unplayed) stats excluded from its own value via shift(1) -- no separate
"carry forward" logic needed, it falls out of the existing computation
for free. This is why round 1 of a new season needs no live data at all
beyond the fixture pairing itself.

Current-season xG is intentionally NOT live-fetched here -- Understat's
season page no longer embeds fetchable JSON the way the historical
scratch/leaguedata_<year>.json snapshots were originally built from
(confirmed by direct inspection: identical, empty page structure for
both a known-good historical season and the current one). Those rows
simply get no xG merged in. This is safe, not a crash risk: pandas' EWM
skips NaN observations rather than propagating them, and the model's
mean-imputation fix already treats a missing covariate as "assume league
average." The practical effect is that Home_xG_Rolling5/Away_xGA_Rolling5
will drift stale over the season (not incorporating this season's actual
xG) until a working live xG source is wired in -- a known follow-up, not
a silent bug.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pipeline"))
from process_season_data import load_data, add_points_per_game, create_team_df  # noqa: E402
from rebuild_rolling_as_ewma import VENUE_ROLLING_COLS, TEAM_ROLLING_COLS  # noqa: E402
from build_full_dataset_no_loss import SEASONS as HISTORICAL_SEASONS, SPAN, load_xg_long  # noqa: E402

from live.fetch_live_elo import fetch_elo_for_dates

RAW_STAT_COLS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]


def _placeholder_raw(fixtures_df):
    """[Date, HomeTeam, AwayTeam] rows -> same shape as load_data()'s
    output, every stat column NaN (this fixture hasn't been played)."""
    df = fixtures_df[["Date", "HomeTeam", "AwayTeam"]].copy()
    df["Time"] = "00:00"
    for c in RAW_STAT_COLS:
        df[c] = float("nan")
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)


def build_training_and_prediction_frames(current_season, current_season_played_df, upcoming_fixtures_df,
                                          train_seasons):
    """
    current_season: e.g. "26-27"
    current_season_played_df: load_data()-shaped DataFrame of this
        season's played matches so far (from fetch_current_season_matches),
        or None if nothing's been published/played yet.
    upcoming_fixtures_df: DataFrame[Date, HomeTeam, AwayTeam] for the
        round about to be predicted (not yet played).
    train_seasons: which HISTORICAL seasons to train on (e.g. the 3 most
        recent complete ones) -- fixed for the whole live season, same as
        WINDOW=3 in every other walk-forward script in this project.

    Returns (train_df, predict_df):
        train_df -- rows for `train_seasons` + current season's
            played-so-far matches, with every covariate + FTHG/FTAG,
            ready for PoissonRegressionGoalsMeanImpute.fit().
        predict_df -- rows for `upcoming_fixtures_df`, same covariate
            columns, ready for .predict_proba().
    """
    all_seasons = list(HISTORICAL_SEASONS) + [current_season]

    raw_frames = {}
    for season in HISTORICAL_SEASONS:
        raw_frames[season] = load_data(f"data/raw/pl{season}.csv")

    current_raw = (
        current_season_played_df if current_season_played_df is not None
        else pd.DataFrame(columns=["Date", "Time", "HomeTeam", "AwayTeam"] + RAW_STAT_COLS)
    )
    placeholder = _placeholder_raw(upcoming_fixtures_df)
    raw_frames[current_season] = (
        pd.concat([current_raw, placeholder], ignore_index=True)
        .sort_values(["Date", "Time"]).reset_index(drop=True)
    )

    # --- PPG, continuous across every season including the live one ---
    ppg_frames = []
    prior_ppg = None
    for season in all_seasons:
        raw = raw_frames[season].copy()
        raw, prior_ppg = add_points_per_game(raw, prior_ppg=prior_ppg)
        raw["Season"] = season
        ppg_frames.append(raw)
    matches = pd.concat(ppg_frames, ignore_index=True)
    matches = matches.rename(columns={"HomePPG": "Home_PPG", "AwayPPG": "Away_PPG", "PPGDiff": "PPG_Difference"})
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")

    # --- Elo: historical from elo_df.csv, current season live-fetched ---
    elo_df = pd.read_csv("data/external/elo_df.csv")
    elo_df["QueryDate"] = pd.to_datetime(elo_df["QueryDate"]).dt.strftime("%Y-%m-%d")

    current_mask = matches["Season"] == current_season
    current_dates = matches.loc[current_mask, "Date_str"].unique().tolist()
    if current_dates:
        # ClubElo only serves ratings for dates that have already happened --
        # fixtures still in the future (the round being predicted) get
        # TODAY's rating instead, since Elo doesn't move without a match
        # being played anyway (same rating a bookmaker would use at kickoff).
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        past_dates = [d for d in current_dates if d <= today_str]
        future_dates = [d for d in current_dates if d > today_str]

        our_teams = set(matches["HomeTeam"]).union(matches["AwayTeam"])
        fetch_dates = set(past_dates) | ({today_str} if future_dates else set())
        live_elo = fetch_elo_for_dates(our_teams, fetch_dates)

        if future_dates and (live_elo["QueryDate"] == today_str).any():
            today_elo = live_elo[live_elo["QueryDate"] == today_str]
            stamped = [today_elo.assign(QueryDate=fd) for fd in future_dates]
            live_elo = pd.concat([live_elo] + stamped, ignore_index=True)

        elo_df = pd.concat([elo_df[~elo_df["QueryDate"].isin(current_dates)], live_elo], ignore_index=True)

    elo_slim = elo_df[["team", "QueryDate", "elo"]]
    matches = matches.merge(
        elo_slim, left_on=["HomeTeam", "Date_str"], right_on=["team", "QueryDate"], how="left"
    ).rename(columns={"elo": "Home_Elo"}).drop(columns=["team", "QueryDate"])
    matches = matches.merge(
        elo_slim, left_on=["AwayTeam", "Date_str"], right_on=["team", "QueryDate"], how="left"
    ).rename(columns={"elo": "Away_Elo"}).drop(columns=["team", "QueryDate"])
    matches["Elo_Difference"] = matches["Home_Elo"] - matches["Away_Elo"]

    # --- Team-centric frame + EWMA rolling features, continuous across
    #     every season including the live/placeholder rows ---
    team_frames = []
    for season in all_seasons:
        raw = raw_frames[season].copy()
        raw["TablePosDiff"] = 0.0  # unused by this model's covariates; create_team_df just needs the column present
        team_df = create_team_df(raw)
        team_frames.append(team_df)
    team_all = pd.concat(team_frames, ignore_index=True)

    team_all["ShotAccuracy"] = team_all["ShotsOnTarget"] / team_all["Shots"].replace(0, 1)
    team_all["GoalDifference"] = team_all["GoalsFor"] - team_all["GoalsAgainst"]
    team_all["ShotDifference"] = team_all["Shots"] - team_all["ShotsAgainst"]
    team_all["ShotOnTargetDifference"] = team_all["ShotsOnTarget"] - team_all["ShotsOnTargetAgainst"]
    team_all["CornerDifference"] = team_all["Corners"] - team_all["CornersAgainst"]
    team_all["FoulDifference"] = team_all["FoulsAgainst"] - team_all["Fouls"]
    team_all["YellowCardDifference"] = team_all["YellowCardsAgainst"] - team_all["YellowCards"]
    team_all["Win"] = team_all["Result"].map({"W": 1, "D": 0.5, "L": 0})
    team_all["Date_str"] = team_all["Date"].dt.strftime("%Y-%m-%d")

    elo_lookup = matches[["Date_str", "Time", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]].copy()
    home_elo_lookup = elo_lookup.rename(columns={"HomeTeam": "Team", "Away_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    away_elo_lookup = elo_lookup.rename(columns={"AwayTeam": "Team", "Home_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    opponent_elo_lookup = pd.concat([home_elo_lookup, away_elo_lookup], ignore_index=True)
    team_all = team_all.merge(opponent_elo_lookup, on=["Date_str", "Time", "Team"], how="left")

    xg_long = load_xg_long()
    team_all = team_all.merge(xg_long, on=["Date_str", "Team"], how="left")

    team_all = team_all.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)

    for col in VENUE_ROLLING_COLS:
        team_all[f"{col}_Rolling5"] = (
            team_all.groupby(["Team", "Venue"])[col]
            .transform(lambda x: x.shift(1).ewm(span=SPAN, min_periods=1).mean())
        )
    for col in TEAM_ROLLING_COLS:
        team_all[f"{col}_RollingTeam7"] = (
            team_all.groupby("Team")[col]
            .transform(lambda x: x.shift(1).ewm(span=SPAN, min_periods=1).mean())
        )

    venue_cols = [f"{c}_Rolling5" for c in VENUE_ROLLING_COLS]
    team_cols = [f"{c}_RollingTeam7" for c in TEAM_ROLLING_COLS]
    all_new = venue_cols + team_cols

    home_roll = (
        team_all[team_all["Venue"] == "H"][["Date", "Time", "Team"] + all_new]
        .rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in all_new}})
    )
    away_roll = (
        team_all[team_all["Venue"] == "A"][["Date", "Time", "Team"] + all_new]
        .rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in all_new}})
    )
    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")
    matches = matches.drop(columns=["Date_str"])

    # --- Split back out: training rows vs the round being predicted ---
    is_placeholder = matches["FTHG"].isna() & (matches["Season"] == current_season)
    predict_df = matches[is_placeholder].copy()

    train_mask = matches["Season"].isin(train_seasons) | (
        (matches["Season"] == current_season) & ~is_placeholder
    )
    train_df = matches[train_mask].copy()

    return train_df, predict_df
