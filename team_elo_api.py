import soccerdata as sd
import pandas as pd
import logging

elo_df = pd.read_csv("elo_df.csv")
all_df = pd.read_csv("dataset/all_seasons.csv")



# make sure both date columns are the same type/format for the merge
all_df["Date"] = pd.to_datetime(all_df["Date"]).dt.strftime("%Y-%m-%d")
elo_df["QueryDate"] = pd.to_datetime(elo_df["QueryDate"]).dt.strftime("%Y-%m-%d")

# keep only what we need from elo_df to avoid column name clutter
elo_slim = elo_df[["team", "QueryDate", "elo"]]

# --- merge for HomeTeam ---
all_df = all_df.merge(
    elo_slim,
    left_on=["HomeTeam", "Date"],
    right_on=["team", "QueryDate"],
    how="left"
)
all_df = all_df.rename(columns={"elo": "Home_Elo"}).drop(columns=["team", "QueryDate"])

# --- merge for AwayTeam ---
all_df = all_df.merge(
    elo_slim,
    left_on=["AwayTeam", "Date"],
    right_on=["team", "QueryDate"],
    how="left"
)
all_df = all_df.rename(columns={"elo": "Away_Elo"}).drop(columns=["team", "QueryDate"])

all_df.to_csv("dataset/all_seasons.csv", index=False)

