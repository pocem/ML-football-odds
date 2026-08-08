"""
Rebuilds the rolling xG/xGA/deep/deep_allowed features using:
  1. The original Understat xG.csv (covers up through ~2024-10-05, all 4 stats)
  2. Manually-collected data pasted by the user for the rest of 24-25
     (xG/xGA only -- no deep/deep_allowed source for this stretch)

Matched to the existing dataset by (Season, HomeTeam, AwayTeam) team-pair
identity rather than date/position, since this project's 24-25 source file
is short of a full 380-match season and positional alignment breaks on
whichever fixtures are missing. Verified 319/319 matches, 0 score
mismatches before this script was written.
"""

import pandas as pd

MATCHES_FILE = "dataset/all_seasons_with_bookies.csv"
XG_FILE = "xG.csv"
PASTE_FILE = "scratch/xg_paste.txt"

TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

TEAMS_2425 = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest",
    "Southampton", "Tottenham", "West Ham", "Wolverhampton Wanderers",
]
TEAMS_SORTED = sorted(TEAMS_2425, key=len, reverse=True)

XG_STATS = ["xG", "xGA", "deep", "deep_allowed"]


def split_two_teams(s):
    for t in TEAMS_SORTED:
        if s.startswith(t):
            rest = s[len(t):]
            if rest in TEAMS_2425:
                return t, rest
    return None, None


def parse_pasted_matches(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    blocks = [b for b in raw.split("\n\n") if b.strip()]

    rows = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        home = lines[0]
        i = 1
        while i < len(lines):
            score_line = lines[i]; i += 1
            xg_line = lines[i]; i += 1
            hs, aws = int(score_line[0]), int(score_line[1])
            xg1, xg2 = float(xg_line[0:4]), float(xg_line[4:8])
            team_line = lines[i]; i += 1
            if team_line in TEAMS_2425:
                away = team_line
                home_next = None
            else:
                away, home_next = split_two_teams(team_line)
            rows.append((home, away, hs, aws, xg1, xg2))
            home = home_next

    df = pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Home_xG", "Away_xG"])
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_MAP)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_MAP)
    return df


def build_understat_raw():
    xg = pd.read_csv(XG_FILE)
    xg["title"] = xg["title"].replace(TEAM_MAP)
    xg["Date"] = pd.to_datetime(xg["date"]).dt.strftime("%Y-%m-%d")
    return xg[["Date", "title", "h_a"] + XG_STATS]


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")

    # --- Source 1: Understat xG.csv, date-based merge (as before) ---
    xg = build_understat_raw()
    home_xg = xg.rename(columns={"title": "HomeTeam", **{c: f"Home_{c}" for c in XG_STATS}}).drop(columns="h_a")
    away_xg = xg.rename(columns={"title": "AwayTeam", **{c: f"Away_{c}" for c in XG_STATS}}).drop(columns="h_a")

    matches = matches.merge(home_xg, left_on=["Date_str", "HomeTeam"], right_on=["Date", "HomeTeam"], how="left", suffixes=("", "_dup"))
    matches = matches.drop(columns=[c for c in matches.columns if c.endswith("_dup")])
    matches = matches.merge(away_xg, left_on=["Date_str", "AwayTeam"], right_on=["Date", "AwayTeam"], how="left", suffixes=("", "_dup"))
    matches = matches.drop(columns=[c for c in matches.columns if c.endswith("_dup")])

    # --- Source 2: pasted 24-25 data, matched by (HomeTeam, AwayTeam) team-pair identity.
    #     MUST also constrain by Season -- a fixture like "Chelsea vs Fulham" recurs across
    #     many different seasons, and without this the pasted 24-25 value would leak into
    #     every other season's occurrence of the same pairing. ---
    pasted = parse_pasted_matches(PASTE_FILE)
    pasted["Season"] = "24-25"
    matches = matches.merge(
        pasted[["Season", "HomeTeam", "AwayTeam", "Home_xG", "Away_xG"]],
        on=["Season", "HomeTeam", "AwayTeam"], how="left", suffixes=("", "_paste")
    )
    matches["Home_xG"] = matches["Home_xG"].fillna(matches["Home_xG_paste"])
    matches["Away_xG"] = matches["Away_xG"].fillna(matches["Away_xG_paste"])
    # xGA is just the opponent's own xG for that match (verified exactly equal,
    # 0.0 max diff across 3610 correctly-paired real fixtures) -- the pasted
    # data only gives xG, so derive xGA from it rather than leaving it NaN.
    matches["Home_xGA"] = matches["Home_xGA"].fillna(matches["Away_xG_paste"])
    matches["Away_xGA"] = matches["Away_xGA"].fillna(matches["Home_xG_paste"])
    matches = matches.drop(columns=["Home_xG_paste", "Away_xG_paste", "Date_str"])

    print("Raw xG coverage by season after combining both sources:")
    print(matches.groupby("Season")["Home_xG"].apply(lambda s: s.notna().sum()))

    # -------------------------------------------------------------------
    # Rebuild rolling features from this combined raw data, same
    # Team+Venue, shift(1)+rolling(5) pattern as everywhere else.
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
    matches = matches.drop(columns=new_roll_cols)  # drop the stale rolling cols before remerging
    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")

    # Drop raw (leaky) per-match stats again -- only the rolling versions are kept.
    raw_cols = [f"Home_{s}" for s in XG_STATS] + [f"Away_{s}" for s in XG_STATS]
    matches = matches.drop(columns=raw_cols)

    print("\nNaN counts in rebuilt rolling columns, by season:")
    nan_mask = matches[new_roll_cols].isna().any(axis=1)
    print(matches[nan_mask].groupby("Season").size())

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
