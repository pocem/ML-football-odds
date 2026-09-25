"""
Fetches ClubElo ratings for specific dates not already cached in
elo_df.csv -- same source/logic as fetch_elo.py / fetch_missing_elo.py.
ClubElo updates daily even between a given team's matches (ratings
drift/adjust based on other leagues too), so re-fetching per prediction
run keeps things using the freshest available rating rather than a stale
end-of-last-season snapshot.

Best-effort: if ClubElo is unreachable for a date, that date's rows fall
back to NaN in the feature build, which the model's mean-imputation fix
already handles (assume league average) rather than crashing.
"""

import pandas as pd
import soccerdata as sd

OUR_TO_CLUBELO = {"Nott'm Forest": "Forest"}
CLUBELO_TO_OUR = {v: k for k, v in OUR_TO_CLUBELO.items()}


def fetch_elo_for_dates(our_teams, dates):
    """dates: iterable of 'YYYY-MM-DD' strings. Returns DataFrame[team, QueryDate, elo].

    soccerdata's ClubElo.read_by_date() already retries internally (5
    attempts with its own growing backoff) before raising -- wrapping
    that in another outer retry loop here just re-runs the whole already
    -exhausted backoff sequence again for no real gain, tripling the
    worst-case wait during a genuine outage. One attempt per date is
    enough; a failure falls back to NaN -> mean-imputed rather than
    blocking the whole run.
    """
    clubelo_teams = {OUR_TO_CLUBELO.get(t, t) for t in our_teams}

    elo = sd.ClubElo()
    frames = []
    for d in sorted(set(dates)):
        try:
            day_df = elo.read_by_date(d).reset_index()
        except Exception as e:
            print(f"  Elo fetch FAILED for {d}, skipping (falls back to NaN -> mean-imputed): {e}")
            continue

        day_df = day_df[day_df["team"].isin(clubelo_teams)].copy()
        day_df["team"] = day_df["team"].replace(CLUBELO_TO_OUR)
        day_df["QueryDate"] = d
        frames.append(day_df[["team", "QueryDate", "elo"]])

    if not frames:
        return pd.DataFrame(columns=["team", "QueryDate", "elo"])
    return pd.concat(frames, ignore_index=True)
