import pandas as pd

MATCHES_FILE = "dataset/all_seasons_with_bookies.csv"

XG_STATS = ["xG", "xGA", "deep", "deep_allowed"]
SEASONS_TO_DROP = ["11-12", "12-13", "13-14"]


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    matches = matches.sort_values(["Date", "Time"]).reset_index(drop=True)

    # -------------------------------------------------------------------
    # Team-centric long view to compute rolling xG stats -- same
    # Team+Venue grouping, shift(1) + rolling(5) pattern as every other
    # Rolling5 feature in this project. Not reset at season boundaries
    # (a team's underlying shot quality doesn't reset on August 1st),
    # matching how Home_OpponentElo_Rolling5 already works.
    # -------------------------------------------------------------------
    home_view = pd.DataFrame({
        "Date": matches["Date"], "Time": matches["Time"],
        "Team": matches["HomeTeam"], "Venue": "H",
        **{stat: matches[f"Home_{stat}"] for stat in XG_STATS},
    })
    away_view = pd.DataFrame({
        "Date": matches["Date"], "Time": matches["Time"],
        "Team": matches["AwayTeam"], "Venue": "A",
        **{stat: matches[f"Away_{stat}"] for stat in XG_STATS},
    })
    team_view = pd.concat([home_view, away_view], ignore_index=True)
    team_view = team_view.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)

    for stat in XG_STATS:
        team_view[f"{stat}_Rolling5"] = (
            team_view
            .groupby(["Team", "Venue"])[stat]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    roll_cols = [f"{stat}_Rolling5" for stat in XG_STATS]

    home_roll = (
        team_view[team_view["Venue"] == "H"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in roll_cols}})
    )
    away_roll = (
        team_view[team_view["Venue"] == "A"][["Date", "Time", "Team"] + roll_cols]
        .rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in roll_cols}})
    )

    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")

    # Drop the raw per-match xG stats -- they leak the current match's own
    # result and were only ever a stepping stone to the rolling versions.
    raw_cols = [f"Home_{s}" for s in XG_STATS] + [f"Away_{s}" for s in XG_STATS]
    matches = matches.drop(columns=raw_cols)

    # Drop the first three seasons entirely, per request.
    before = len(matches)
    matches = matches[~matches["Season"].isin(SEASONS_TO_DROP)].reset_index(drop=True)
    print(f"Dropped {before - len(matches)} rows from seasons {SEASONS_TO_DROP}")

    new_roll_cols = [f"Home_{c}" for c in roll_cols] + [f"Away_{c}" for c in roll_cols]
    print("\nNaN counts in new rolling columns (expected: only each team's very first match at that venue):")
    print(matches[new_roll_cols].isna().sum())

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
