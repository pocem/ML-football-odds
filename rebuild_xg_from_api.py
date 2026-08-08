"""
Rebuilds Home/Away xG, xGA, deep, deep_allowed Rolling5 features from
Understat's live getLeagueData JSON endpoint (fetched for every season
2014-2025, saved under scratch/leaguedata_<year>.json), replacing the
earlier CSV + manually-pasted-text patchwork with one complete, consistent
source -- every season now has full 760/760 team-match coverage.
"""

import json

import pandas as pd

MATCHES_FILE = "dataset/all_seasons_with_bookies.csv"
SCRATCH_DIR = "scratch"
YEARS = range(2014, 2026)

TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

XG_STATS = ["xG", "xGA", "deep", "deep_allowed"]


def load_all_years():
    rows = []
    for year in YEARS:
        with open(f"{SCRATCH_DIR}/leaguedata_{year}.json", encoding="utf-8") as f:
            data = json.load(f)
        for team in data["teams"].values():
            title = TEAM_MAP.get(team["title"], team["title"])
            for match in team["history"]:
                rows.append({
                    "Date": pd.to_datetime(match["date"]).strftime("%Y-%m-%d"),
                    "title": title,
                    "h_a": match["h_a"],
                    "xG": match["xG"],
                    "xGA": match["xGA"],
                    "deep": match["deep"],
                    "deep_allowed": match["deep_allowed"],
                })
    return pd.DataFrame(rows)


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")

    raw = load_all_years()

    # --- Sanity check the merge key before trusting it: h_a should agree
    #     with whether the team was actually Home or Away in our data. ---
    home_check = matches.merge(raw, left_on=["Date_str", "HomeTeam"], right_on=["Date", "title"], how="inner")
    away_check = matches.merge(raw, left_on=["Date_str", "AwayTeam"], right_on=["Date", "title"], how="inner")
    assert (home_check["h_a"] == "h").all(), "Home-side merge produced a non-'h' row -- alignment is wrong"
    assert (away_check["h_a"] == "a").all(), "Away-side merge produced a non-'a' row -- alignment is wrong"
    print(f"Sanity check passed: {len(home_check)} home rows all 'h', {len(away_check)} away rows all 'a'")

    home_xg = raw.rename(columns={"title": "HomeTeam", **{c: f"Home_{c}" for c in XG_STATS}}).drop(columns=["h_a"])
    away_xg = raw.rename(columns={"title": "AwayTeam", **{c: f"Away_{c}" for c in XG_STATS}}).drop(columns=["h_a"])

    matches = matches.merge(home_xg, left_on=["Date_str", "HomeTeam"], right_on=["Date", "HomeTeam"], how="left", suffixes=("", "_dup"))
    matches = matches.drop(columns=[c for c in matches.columns if c.endswith("_dup")])
    matches = matches.merge(away_xg, left_on=["Date_str", "AwayTeam"], right_on=["Date", "AwayTeam"], how="left", suffixes=("", "_dup"))
    matches = matches.drop(columns=[c for c in matches.columns if c.endswith("_dup")])
    matches = matches.drop(columns=["Date_str"])

    print("\nRaw xG coverage by season:")
    print(matches.groupby("Season")["Home_xG"].apply(lambda s: s.notna().sum()))

    # -------------------------------------------------------------------
    # Rolling features, same Team+Venue, shift(1)+rolling(5) pattern as
    # everywhere else. Not reset at season boundaries.
    # -------------------------------------------------------------------
    matches = matches.sort_values(["Date", "Time"]).reset_index(drop=True)

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
            team_view.groupby(["Team", "Venue"])[stat]
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

    new_roll_cols = [f"Home_{c}" for c in roll_cols] + [f"Away_{c}" for c in roll_cols]
    matches = matches.drop(columns=[c for c in new_roll_cols if c in matches.columns])
    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")

    raw_cols = [f"Home_{s}" for s in XG_STATS] + [f"Away_{s}" for s in XG_STATS]
    matches = matches.drop(columns=raw_cols)

    print("\nNaN counts per rolling column, by season:")
    print(matches.groupby("Season")[new_roll_cols].apply(lambda g: g.isna().sum().sum()))

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
