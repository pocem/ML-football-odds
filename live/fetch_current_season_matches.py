"""
Fetches the current season's played-match stats (shots, corners, fouls,
cards, scores) from football-data.co.uk -- same source/format as every
historical pl{season}.csv file, so it loads via the existing
process_season_data.load_data() unmodified. Returns None (not an error)
if the file isn't published yet -- normal before a season's first match;
football-data.co.uk simply doesn't have the file up until then.
"""

import os
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pipeline"))
from process_season_data import load_data  # noqa: E402

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
CACHE_DIR = "data/raw"


def _season_code(season_str):
    """'26-27' -> '2627'"""
    a, b = season_str.split("-")
    return f"{a}{b}"


def fetch_current_season_matches(season_str):
    code = _season_code(season_str)
    url = FD_URL.format(code=code)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    except requests.RequestException as e:
        print(f"Could not reach {url}: {e}")
        return None

    if resp.status_code != 200 or "HomeTeam" not in resp.text[:1000]:
        print(f"No published match file yet for {season_str} at {url} "
              f"(normal if the season hasn't started, or no matches played yet)")
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/pl{season_str}_live.csv"
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    df = load_data(cache_path)
    print(f"Fetched {len(df)} played matches for {season_str} from {url}")
    return df


if __name__ == "__main__":
    df = fetch_current_season_matches("26-27")
    print(df)
