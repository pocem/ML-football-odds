"""
Fetches the full-season fixture schedule (all 38 rounds, played and
unplayed) from fixturedownload.com's no-auth JSON feed -- the only source
of round numbers and kickoff dates/pairings used here. It does NOT carry
full match stats (shots, corners, etc.), only the final score once a
match has been played -- full stats for played matches come from
football-data.co.uk instead (fetch_current_season_matches.py).

No API key, no registration -- verified directly:
https://fixturedownload.com/feed/json/epl-2026 returns the full 380-match
2026-27 schedule as plain JSON.
"""

import pandas as pd
import requests

from live.team_map import normalize_team

FIXTURE_FEED_URL = "https://fixturedownload.com/feed/json/{season_slug}"


def fetch_season_fixtures(season_slug="epl-2026"):
    url = FIXTURE_FEED_URL.format(season_slug=season_slug)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["DateUtc"]).dt.tz_localize(None)
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)
    df["Played"] = df["HomeTeamScore"].notna() & df["AwayTeamScore"].notna()

    return df[["RoundNumber", "Date", "HomeTeam", "AwayTeam",
               "HomeTeamScore", "AwayTeamScore", "Played"]].sort_values(
        ["RoundNumber", "Date"]).reset_index(drop=True)


def next_unplayed_round(fixtures_df):
    """Smallest RoundNumber with at least one unplayed fixture, or None if
    the whole season is done."""
    unplayed = fixtures_df[~fixtures_df["Played"]]
    if unplayed.empty:
        return None
    return int(unplayed["RoundNumber"].min())


if __name__ == "__main__":
    df = fetch_season_fixtures()
    print(f"Fetched {len(df)} fixtures, {df['Played'].sum()} played")
    rnd = next_unplayed_round(df)
    print(f"Next round to predict: {rnd}")
    print(df[df["RoundNumber"] == rnd])
