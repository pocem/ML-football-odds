"""
Team-name normalization between fixturedownload.com's naming and this
project's canonical naming (football-data.co.uk convention, same as
every pl{season}.csv file). Verified by diffing fixturedownload.com's
2026-27 team list against data/processed/all_seasons_14window_ppg_full.csv's
team list directly -- every other team name already matches exactly.
Coventry is newly promoted and genuinely absent from our historical data
(not a naming mismatch) -- left unmapped on purpose, so it falls through
to the model's mean-imputation path for a team with no history.
"""

FIXTUREDOWNLOAD_TO_CANONICAL = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

# Tipsport.sk odds are pasted in manually (see market_odds_paste.txt /
# merge_market_odds.py) -- their naming also differs slightly from ours.
TIPSPORT_TO_CANONICAL = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham": "Nott'm Forest",
}


def normalize_team(name):
    return FIXTUREDOWNLOAD_TO_CANONICAL.get(name, name)


def normalize_tipsport_team(name):
    return TIPSPORT_TO_CANONICAL.get(name, name)
