import pandas as pd

MATCHES_FILE = "dataset/all_seasons_with_bookies.csv"
XG_FILE = "xG.csv"

# Understat's team names vs. this project's football-data.co.uk naming.
# Teams that only appear on our side (Blackburn, Bolton, Reading, Wigan) were
# out of the Premier League before Understat's coverage starts (2014-15) --
# not a mapping issue, just outside the xG data's window.
TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

XG_COLS = ["xG", "xGA", "deep", "deep_allowed"]


def load_xg():
    xg = pd.read_csv(XG_FILE)
    xg["title"] = xg["title"].replace(TEAM_MAP)
    xg["Date"] = pd.to_datetime(xg["date"]).dt.strftime("%Y-%m-%d")
    return xg[["Date", "title", "h_a"] + XG_COLS]


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    matches["Date"] = matches["Date"].dt.strftime("%Y-%m-%d")

    xg = load_xg()

    home_xg = xg.rename(columns={"title": "HomeTeam", **{c: f"Home_{c}" for c in XG_COLS}}).drop(columns="h_a")
    away_xg = xg.rename(columns={"title": "AwayTeam", **{c: f"Away_{c}" for c in XG_COLS}}).drop(columns="h_a")

    matches = matches.merge(home_xg, on=["Date", "HomeTeam"], how="left")
    matches = matches.merge(away_xg, on=["Date", "AwayTeam"], how="left")

    matches["Date"] = pd.to_datetime(matches["Date"])

    new_cols = [f"Home_{c}" for c in XG_COLS] + [f"Away_{c}" for c in XG_COLS]
    covered = matches[new_cols[0]].notna().sum()
    print(f"Matches with xG coverage: {covered} / {len(matches)} ({covered / len(matches):.1%})")

    print("\nCoverage by season:")
    coverage_by_season = matches.groupby("Season")[new_cols[0]].apply(lambda s: s.notna().sum())
    total_by_season = matches.groupby("Season").size()
    print(pd.DataFrame({"matched": coverage_by_season, "total": total_by_season}))

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
