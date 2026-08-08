"""
Builds dataset/all_seasons_14window.csv -- same as all_seasons_with_bookies.csv,
but with every EWMA rolling feature (both the venue-split _Rolling5 family and
the team-overall _RollingTeam7 family) recomputed at span=14, the value found
by models/Poisson_Bivariate_intraseason_rolling_window_sweep.py's convergence
sweep. Column names are left unchanged (still _Rolling5/_RollingTeam7) so every
existing model script can point DATA_PATH at this file without any other edits.

Doesn't touch all_seasons_with_bookies.csv (currently span=10) -- writes to a
separate file so both are available side by side.
"""

import pandas as pd

from rebuild_rolling_as_ewma import (
    build_team_centric,
    VENUE_ROLLING_COLS,
    TEAM_ROLLING_COLS,
)

INPUT_FILE = "dataset/all_seasons_with_bookies.csv"
OUTPUT_FILE = "dataset/all_seasons_14window.csv"
SPAN = 14


def main():
    matches = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    seasons = sorted(matches["Season"].unique())

    old_rolling_cols = [c for c in matches.columns if c.endswith("_Rolling5") or "RollingTeam7" in c]
    print(f"Dropping {len(old_rolling_cols)} old rolling columns")
    matches = matches.drop(columns=old_rolling_cols)

    # Home_TablePosDiff/Away_TablePosDiff (the BASE, non-rolling columns) are
    # stale leftovers from whatever originally built all_seasons.csv, predating
    # the add_table_positions() carryover fix -- drop and rebuild those too,
    # not just the rolling versions, or they'd silently stay on the old flat-10
    # early-season logic while everything else gets the fix.
    print("Dropping stale base Home_TablePosDiff/Away_TablePosDiff columns")
    matches = matches.drop(columns=["Home_TablePosDiff", "Away_TablePosDiff"])

    team_all = build_team_centric(matches, seasons)

    base_home = (
        team_all[team_all["Venue"] == "H"][["Date", "Time", "Team", "TablePosDiff"]]
        .rename(columns={"Team": "HomeTeam", "TablePosDiff": "Home_TablePosDiff"})
    )
    base_away = (
        team_all[team_all["Venue"] == "A"][["Date", "Time", "Team", "TablePosDiff"]]
        .rename(columns={"Team": "AwayTeam", "TablePosDiff": "Away_TablePosDiff"})
    )
    matches = matches.merge(base_home, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(base_away, on=["Date", "Time", "AwayTeam"], how="left")

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

    new_cols = [f"Home_{c}" for c in all_new] + [f"Away_{c}" for c in all_new]
    print(f"Added {len(new_cols)} EWMA columns at span={SPAN}")
    nan_mask = matches[new_cols].isna().any(axis=1)
    print("NaN counts by season (rows with any NaN among new columns):")
    print(matches[nan_mask].groupby("Season").size())

    matches.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {matches.shape} to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
