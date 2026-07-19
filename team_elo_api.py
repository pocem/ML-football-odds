import pandas as pd

MATCHES_FILE = "dataset/all_seasons.csv"
ELO_FILE = "elo_df.csv"


def add_elo_features():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    elo_df = pd.read_csv(ELO_FILE)

    matches["Date"] = matches["Date"].dt.strftime("%Y-%m-%d")
    elo_df["QueryDate"] = pd.to_datetime(elo_df["QueryDate"]).dt.strftime("%Y-%m-%d")
    elo_slim = elo_df[["team", "QueryDate", "elo"]]

    # --- Home / Away Elo ---
    matches = matches.merge(
        elo_slim, left_on=["HomeTeam", "Date"], right_on=["team", "QueryDate"], how="left"
    ).rename(columns={"elo": "Home_Elo"}).drop(columns=["team", "QueryDate"])

    matches = matches.merge(
        elo_slim, left_on=["AwayTeam", "Date"], right_on=["team", "QueryDate"], how="left"
    ).rename(columns={"elo": "Away_Elo"}).drop(columns=["team", "QueryDate"])

    matches["Elo_Difference"] = matches["Home_Elo"] - matches["Away_Elo"]

    matches["Date"] = pd.to_datetime(matches["Date"])
    matches = matches.sort_values(["Date", "Time"]).reset_index(drop=True)

    # -------------------------------------------------
    # Rolling5 opponent Elo (strength of schedule),
    # grouped by Team + Venue to match every other Rolling5
    # feature already in this dataset. Shifted by 1 so the
    # current match's own opponent never leaks into its own average.
    # -------------------------------------------------

    home_view = pd.DataFrame({
        "Date": matches["Date"],
        "Time": matches["Time"],
        "Team": matches["HomeTeam"],
        "Venue": "H",
        "OpponentElo": matches["Away_Elo"],
    })
    away_view = pd.DataFrame({
        "Date": matches["Date"],
        "Time": matches["Time"],
        "Team": matches["AwayTeam"],
        "Venue": "A",
        "OpponentElo": matches["Home_Elo"],
    })

    team_view = pd.concat([home_view, away_view], ignore_index=True)
    team_view = team_view.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)

    team_view["OpponentElo_Rolling5"] = (
        team_view
        .groupby(["Team", "Venue"])["OpponentElo"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    home_roll = (
        team_view[team_view["Venue"] == "H"][["Date", "Time", "Team", "OpponentElo_Rolling5"]]
        .rename(columns={"Team": "HomeTeam", "OpponentElo_Rolling5": "Home_OpponentElo_Rolling5"})
    )
    away_roll = (
        team_view[team_view["Venue"] == "A"][["Date", "Time", "Team", "OpponentElo_Rolling5"]]
        .rename(columns={"Team": "AwayTeam", "OpponentElo_Rolling5": "Away_OpponentElo_Rolling5"})
    )

    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")

    matches.to_csv(MATCHES_FILE, index=False)

    print(f"Saved {matches.shape} to {MATCHES_FILE}")
    print(f"Missing Home_Elo: {matches['Home_Elo'].isna().sum()}")
    print(f"Missing Away_Elo: {matches['Away_Elo'].isna().sum()}")


if __name__ == "__main__":
    add_elo_features()
