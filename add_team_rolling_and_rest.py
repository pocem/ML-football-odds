"""
Adds "team-overall" Rolling7 features (suffix _RollingTeam7) to
all_seasons_with_bookies.csv -- grouped by Team only, not Team+Venue.
Every existing _Rolling5 feature only looks at a team's last 5 HOME (or
AWAY) matches, which can span 10+ real match-weeks before enough home
games accumulate. These instead use the team's actual last 7 matches
regardless of venue -- true recency.

(RestDays was tried and removed -- with only Premier League fixture data,
it can't see Champions League/cup matches, so a team playing a midweek
European tie would look "well rested" in PL-only date gaps. Misleading
rather than just incomplete, so dropped rather than kept as a known-bad
feature.)

Rebuilt from the raw per-match stats (the existing dataset only kept the
already-venue-rolled outputs, not the raw shots/corners/fouls feeding
them) plus the Understat xG data already fetched into scratch/.
"""

import json

import pandas as pd

from process_season_data import load_data, add_table_positions, create_team_df

MATCHES_FILE = "dataset/all_seasons_with_bookies.csv"
SCRATCH_DIR = "scratch"
WINDOW = 7

TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

CORE_ROLLING_COLS = [
    "TablePosDiff", "GoalsFor", "GoalsAgainst", "GoalDifference",
    "Shots", "ShotsAgainst", "ShotDifference",
    "ShotsOnTarget", "ShotsOnTargetAgainst", "ShotOnTargetDifference",
    "ShotAccuracy", "Corners", "CornerDifference", "Fouls", "FoulDifference",
    "Points", "Win",
]
XG_STATS = ["xG", "xGA", "deep", "deep_allowed"]
ALL_ROLLING_COLS = CORE_ROLLING_COLS + XG_STATS


def build_team_centric_all_seasons(seasons):
    team_frames = []
    prior_positions = None
    for season in seasons:
        raw = load_data(f"pl{season}.csv")
        raw, prior_positions = add_table_positions(raw, prior_positions=prior_positions)
        team_df = create_team_df(raw)
        team_frames.append(team_df)
    team_all = pd.concat(team_frames, ignore_index=True)

    # Same derived stats add_rolling_features() computes, minus the rolling step itself.
    team_all["ShotAccuracy"] = team_all["ShotsOnTarget"] / team_all["Shots"].replace(0, 1)
    team_all["GoalDifference"] = team_all["GoalsFor"] - team_all["GoalsAgainst"]
    team_all["ShotDifference"] = team_all["Shots"] - team_all["ShotsAgainst"]
    team_all["ShotOnTargetDifference"] = team_all["ShotsOnTarget"] - team_all["ShotsOnTargetAgainst"]
    team_all["CornerDifference"] = team_all["Corners"] - team_all["CornersAgainst"]
    team_all["FoulDifference"] = team_all["FoulsAgainst"] - team_all["Fouls"]
    team_all["Win"] = team_all["Result"].map({"W": 1, "D": 0.5, "L": 0})

    return team_all


def load_xg_long():
    rows = []
    for year in range(2014, 2026):
        with open(f"{SCRATCH_DIR}/leaguedata_{year}.json", encoding="utf-8") as f:
            data = json.load(f)
        for team in data["teams"].values():
            title = TEAM_MAP.get(team["title"], team["title"])
            for m in team["history"]:
                rows.append({
                    "Date": pd.to_datetime(m["date"]).strftime("%Y-%m-%d"),
                    "Team": title,
                    "xG": m["xG"], "xGA": m["xGA"],
                    "deep": m["deep"], "deep_allowed": m["deep_allowed"],
                })
    return pd.DataFrame(rows)


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])

    # Drop columns from earlier runs of this script (old _RollingTeam5 window,
    # and the now-removed RestDays features) so nothing stale is left behind.
    stale_cols = [c for c in matches.columns if "RollingTeam" in c or "RestDays" in c]
    if stale_cols:
        print(f"Dropping {len(stale_cols)} stale columns from a previous run: {stale_cols}")
        matches = matches.drop(columns=stale_cols)

    seasons = sorted(matches["Season"].unique())

    team_all = build_team_centric_all_seasons(seasons)
    team_all["Date_str"] = team_all["Date"].dt.strftime("%Y-%m-%d")

    xg_long = load_xg_long()
    team_all = team_all.merge(
        xg_long, left_on=["Date_str", "Team"], right_on=["Date", "Team"],
        how="left", suffixes=("", "_xg")
    )
    team_all = team_all.drop(columns=["Date_xg"])

    xg_missing = team_all["xG"].isna().sum()
    print(f"Team-match rows missing xG after merge: {xg_missing} / {len(team_all)}")

    team_all = team_all.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)

    # -------------------------------------------------------------------
    # Rolling(7), grouped by TEAM ONLY -- true recency, not reset at
    # season boundaries (matches how Elo/xG rolling already work here).
    # -------------------------------------------------------------------
    for col in ALL_ROLLING_COLS:
        team_all[f"{col}_RollingTeam{WINDOW}"] = (
            team_all.groupby("Team")[col]
            .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
        )

    roll_cols = [f"{col}_RollingTeam{WINDOW}" for col in ALL_ROLLING_COLS]

    home_roll = (
        team_all[team_all["Venue"] == "H"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in roll_cols}})
    )
    away_roll = (
        team_all[team_all["Venue"] == "A"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in roll_cols}})
    )

    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")

    new_cols = [f"Home_{c}" for c in roll_cols] + [f"Away_{c}" for c in roll_cols]
    print(f"\nAdded {len(new_cols)} new columns")
    print("\nNaN counts by season (rows with any NaN among new columns):")
    nan_mask = matches[new_cols].isna().any(axis=1)
    print(matches[nan_mask].groupby("Season").size())

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
