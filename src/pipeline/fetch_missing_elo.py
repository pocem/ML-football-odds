"""
Fetches ClubElo ratings for match dates that elo_df.csv is missing.

fetch_elo.py originally built its date list from data/processed/all_seasons.csv,
which -- via process_season_data.export_dataset()'s per-season dropna() --
never contained any of the ~245 season-opening matches later recovered by
build_full_dataset_no_loss.py. Their dates were therefore never queried
at all (confirmed: every currently-missing Home_Elo/Away_Elo date is
completely absent from elo_df.csv, not a per-team gap on a date that WAS
queried). This just fetches those specific missing dates and appends them
to the existing elo_df.csv -- same fetch logic as fetch_elo.py.
"""

import time

import pandas as pd
import soccerdata as sd

OUR_TO_CLUBELO = {"Nott'm Forest": "Forest"}
CLUBELO_TO_OUR = {v: k for k, v in OUR_TO_CLUBELO.items()}

FULL_DATASET = "data/processed/all_seasons_14window_ppg_full.csv"
ELO_FILE = "data/external/elo_df.csv"


def main():
    new = pd.read_csv(FULL_DATASET, parse_dates=["Date"])
    elo_df = pd.read_csv(ELO_FILE)
    elo_dates = set(pd.to_datetime(elo_df["QueryDate"]).dt.strftime("%Y-%m-%d"))

    missing_rows = new[new["Home_Elo"].isna() | new["Away_Elo"].isna()]
    missing_dates = sorted(set(missing_rows["Date"].dt.strftime("%Y-%m-%d")) - elo_dates)
    print(f"Fetching Elo for {len(missing_dates)} previously-unqueried dates...")

    our_teams = set(new["HomeTeam"]).union(new["AwayTeam"])
    clubelo_teams = {OUR_TO_CLUBELO.get(t, t) for t in our_teams}

    elo = sd.ClubElo()
    frames = []
    for i, date in enumerate(missing_dates):
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
        print(f"  {i + 1}/{len(missing_dates)}: {date} -> {len(day_df)} teams")

    new_elo = pd.concat(frames, ignore_index=True)
    combined = pd.concat([elo_df, new_elo], ignore_index=True)
    combined.to_csv(ELO_FILE, index=False)
    print(f"\nAppended {new_elo.shape[0]} rows. Saved {combined.shape} to {ELO_FILE}")


if __name__ == "__main__":
    main()
