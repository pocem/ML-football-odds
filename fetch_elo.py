import time

import pandas as pd
import soccerdata as sd

# ClubElo uses "Forest" where our match data uses "Nott'm Forest".
# Every other Premier League team name in this dataset already matches
# ClubElo's naming (verified by spot-checking several seasons).
OUR_TO_CLUBELO = {"Nott'm Forest": "Forest"}
CLUBELO_TO_OUR = {v: k for k, v in OUR_TO_CLUBELO.items()}

INPUT_FILE = "dataset/all_seasons.csv"
OUTPUT_FILE = "elo_df.csv"


def fetch_all_elo():
    matches = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    our_teams = set(matches["HomeTeam"]).union(matches["AwayTeam"])
    clubelo_teams = {OUR_TO_CLUBELO.get(t, t) for t in our_teams}

    dates = sorted(matches["Date"].dt.strftime("%Y-%m-%d").unique())
    print(f"Fetching Elo for {len(dates)} unique dates...")

    elo = sd.ClubElo()
    frames = []
    for i, date in enumerate(dates):
        for attempt in range(3):
            try:
                day_df = elo.read_by_date(date).reset_index()
                break
            except Exception as e:
                print(f"  retry {date} ({attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            print(f"  FAILED {date}, skipping")
            continue

        day_df = day_df[day_df["team"].isin(clubelo_teams)].copy()
        day_df["team"] = day_df["team"].replace(CLUBELO_TO_OUR)
        day_df["QueryDate"] = date
        frames.append(day_df)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dates)} dates done")

    elo_df = pd.concat(frames, ignore_index=True)
    elo_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {elo_df.shape} to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_all_elo()
