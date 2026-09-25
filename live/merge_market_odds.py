"""
Merges manually-pasted Tipsport odds (live/market_odds_paste.txt, see
parse_tipsport_odds.py) into predictions_readable.csv and
predictions_log.csv, matched by (home_team, away_team) within the most
recent round in each log. Safe to re-run -- it overwrites the
market_fair_odds_* columns for matched rows rather than appending
duplicates.

Only de-vigged ("fair") market odds are stored -- raw bookmaker odds are
parsed from the paste (needed to compute the de-vig) but never written to
either CSV. Raw odds carry a margin (vig), so 1/odds across the three
outcomes sums to more than 1.0 (Tipsport's 1X2 book here runs about 3-5%
over); comparing that directly against the model's own probabilities
(which do sum to exactly 1.0) isn't a fair fight, and there's no reason
to keep the raw numbers around once the fair ones are computed. De-vigging
uses the standard proportional method (same approach the rest of this
project already uses for Bet365/avg-bookie odds):
    fair_p_i = (1/odds_i) / sum_j(1/odds_j)

Usage: paste a fresh Tipsport match list into live/market_odds_paste.txt,
then run this file.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live.parse_tipsport_odds import parse_tipsport_file  # noqa: E402
from live.build_analysis_workbook import main as rebuild_analysis_workbook, OUTPUT_XLSX  # noqa: E402

PASTE_FILE = "live/market_odds_paste.txt"
LOG_CSV = "live/predictions_log.csv"
READABLE_LOG_CSV = "live/predictions_readable.csv"


def devig_row(home_odds, draw_odds, away_odds):
    """Proportional de-vig. Returns (fair_odds_home, fair_odds_draw, fair_odds_away)."""
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return pd.NA, pd.NA, pd.NA
    inv = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    overround = sum(inv)
    fair_p = [x / overround for x in inv]
    return tuple(round(1 / p, 2) for p in fair_p)


def merge_into(path, market_odds):
    df = pd.read_csv(path)
    latest_round = df["round"].max()
    mask = df["round"] == latest_round

    for col in ["market_fair_odds_home", "market_fair_odds_draw", "market_fair_odds_away"]:
        if col not in df.columns:
            df[col] = pd.NA

    merged_count = 0
    for _, row in market_odds.iterrows():
        row_mask = mask & (df["home_team"] == row["HomeTeam"]) & (df["away_team"] == row["AwayTeam"])
        if not row_mask.any():
            print(f"  No matching row in {path} for {row['HomeTeam']} vs {row['AwayTeam']} (round {latest_round})")
            continue
        fair_h, fair_d, fair_a = devig_row(
            row["market_odds_home"], row["market_odds_draw"], row["market_odds_away"]
        )
        df.loc[row_mask, "market_fair_odds_home"] = fair_h
        df.loc[row_mask, "market_fair_odds_draw"] = fair_d
        df.loc[row_mask, "market_fair_odds_away"] = fair_a
        merged_count += 1

    df.to_csv(path, index=False)
    print(f"Merged {merged_count}/{len(market_odds)} matches into {path}")


def main():
    market_odds = parse_tipsport_file(PASTE_FILE)
    print(f"Parsed {len(market_odds)} matches from {PASTE_FILE}")

    merge_into(READABLE_LOG_CSV, market_odds)
    merge_into(LOG_CSV, market_odds)

    try:
        rebuild_analysis_workbook()
    except PermissionError:
        print(f"\nCould not rebuild {OUTPUT_XLSX} -- it's probably still open in Excel. "
              f"Close it and re-run live/build_analysis_workbook.py to refresh it.")


if __name__ == "__main__":
    main()
