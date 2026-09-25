"""
Parses odds copy-pasted (by hand, in a browser) from a Tipsport.sk match
list into a clean DataFrame. Manual paste, not scraping -- Tipsport's
terms of service explicitly prohibit automated extraction of their odds
data, so this only ever reads a text file you've pasted into yourself.

Tipsport's copy-paste dump has 5 numbers per match (1X2 plus two double
-chance markets we don't want): "1" (home), "10" (home-or-draw), "0"
(draw), "02" (draw-or-away), "2" (away) -- in that column order. We keep
only the 1st, 3rd and 5th (the plain 1X2 odds) and discard the double
-chance pair.

If a match's odds got cut off in the paste (page didn't finish loading
before copying, e.g.), that match is skipped with a warning rather than
guessed at or misaligned onto the next match.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live.team_map import normalize_tipsport_team  # noqa: E402

_HEADER_SEQUENCE = ["1", "10", "0", "02", "2"]


def parse_tipsport_paste(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    # Skip past the "1 / 10 / 0 / 02 / 2" column-header block (and
    # whatever precedes it, e.g. the league name) by matching this exact
    # known token sequence -- not by guessing from line position/shape,
    # since a league name can itself contain " - " (e.g. "Futbal - muzi")
    # and false-match a positional heuristic.
    i = 0
    for j in range(len(lines) - len(_HEADER_SEQUENCE) + 1):
        if lines[j:j + len(_HEADER_SEQUENCE)] == _HEADER_SEQUENCE:
            i = j + len(_HEADER_SEQUENCE)
            break

    matches = []
    while i < len(lines):
        if " - " not in lines[i]:
            i += 1
            continue

        home_raw, away_raw = (t.strip() for t in lines[i].split(" - ", 1))
        home, away = normalize_tipsport_team(home_raw), normalize_tipsport_team(away_raw)
        i += 1

        if i < len(lines):  # date/time line
            i += 1
        if i < len(lines) and lines[i].startswith("+"):  # "+90"-style market-count line
            i += 1

        odds = []
        while i < len(lines) and len(odds) < 5:
            try:
                odds.append(float(lines[i]))
            except ValueError:
                break
            i += 1

        if len(odds) == 5:
            matches.append({
                "HomeTeam": home, "AwayTeam": away,
                "market_odds_home": odds[0],
                "market_odds_draw": odds[2],
                "market_odds_away": odds[4],
            })
        else:
            print(f"Skipping {home} - {away}: only found {len(odds)}/5 odds values (paste likely cut off)")

    return pd.DataFrame(matches)


def parse_tipsport_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_tipsport_paste(f.read())


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "live/market_odds_paste.txt"
    df = parse_tipsport_file(path)
    print(df.to_string())
