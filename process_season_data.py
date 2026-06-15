import pandas as pd

def load_data(path):
    # Explicitly define the exact columns you want to keep
    keep_columns = [
        'Date', 'Time', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 
        'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
    ]
    
    # Read ONLY these columns from the CSV file
    df = pd.read_csv(path, usecols=keep_columns)
    
    # Standard formatting
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)
    
    return df

def create_team_df(df):
    ### Team-centric dataframe, extra column for points ###
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)
    if 'TablePosDiff' not in df.columns:
        raise ValueError("df is missing 'TablePosDiff' — make sure to call add_table_positions(df) first.")
    # -------------------------
    # HOME TEAM PERSPECTIVE
    # -------------------------

    home_df = pd.DataFrame({
        'Date': df['Date'],
        'Time': df['Time'],
        'Team': df['HomeTeam'],
        'Opponent': df['AwayTeam'],
        'Venue': 'H',
        
        # Pull TablePosDiff from the main match dataframe
        'TablePosDiff': df['TablePosDiff'],

        'GoalsFor': df['FTHG'],
        'GoalsAgainst': df['FTAG'],

        'HTGoalsFor': df['HTHG'],
        'HTGoalsAgainst': df['HTAG'],

        'Shots': df['HS'],
        'ShotsAgainst': df['AS'],

        'ShotsOnTarget': df['HST'],
        'ShotsOnTargetAgainst': df['AST'],

        'Corners': df['HC'],
        'CornersAgainst': df['AC'],

        'Fouls': df['HF'],
        'FoulsAgainst': df['AF'],

        'YellowCards': df['HY'],
        'YellowCardsAgainst': df['AY'],

        'RedCards': df['HR'],
        'RedCardsAgainst': df['AR']
    })

    # Result from home team's perspective
    home_df['Result'] = df['FTR'].map({
        'H': 'W',
        'D': 'D',
        'A': 'L'
    })

    # -------------------------
    # AWAY TEAM PERSPECTIVE
    # -------------------------

    away_df = pd.DataFrame({
        'Date': df['Date'],
        'Time': df['Time'],
        'Team': df['AwayTeam'],
        'Opponent': df['HomeTeam'],
        'Venue': 'A',
        
        # Invert the difference for the away team's perspective
        'TablePosDiff': -df['TablePosDiff'],

        'GoalsFor': df['FTAG'],
        'GoalsAgainst': df['FTHG'],

        'HTGoalsFor': df['HTAG'],
        'HTGoalsAgainst': df['HTHG'],

        'Shots': df['AS'],
        'ShotsAgainst': df['HS'],

        'ShotsOnTarget': df['AST'],
        'ShotsOnTargetAgainst': df['HST'],

        'Corners': df['AC'],
        'CornersAgainst': df['HC'],

        'Fouls': df['AF'],
        'FoulsAgainst': df['HF'],

        'YellowCards': df['AY'],
        'YellowCardsAgainst': df['HY'],

        'RedCards': df['AR'],
        'RedCardsAgainst': df['HR']
    })

    # Result from away team's perspective
    away_df['Result'] = df['FTR'].map({
        'A': 'W',
        'D': 'D',
        'H': 'L'
    })

    # -------------------------
    # COMBINE
    # -------------------------

    team_df = pd.concat([home_df, away_df], ignore_index=True)

    # Points column
    team_df['Points'] = team_df['Result'].map({
        'W': 3,
        'D': 1,
        'L': 0
    })

    # Sort chronologically
    team_df = team_df.sort_values(
        ['Team', 'Date', 'Time']
    ).reset_index(drop=True)

    return team_df

def add_table_positions(df):
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    teams = sorted(
        set(df['HomeTeam']).union(set(df['AwayTeam']))
    )

    table = pd.DataFrame(index=teams)
    table['Pts'] = 0
    table['GD'] = 0
    table['GF'] = 0

    played = {team: 0 for team in teams}

    home_pos = []
    away_pos = []

    for _, row in df.iterrows():

        home = row['HomeTeam']
        away = row['AwayTeam']

        standings = (
            table
            .sort_values(['Pts', 'GD', 'GF'], ascending=False)
            .reset_index()
        )

        standings['Position'] = standings.index + 1
        pos_lookup = dict(zip(standings['index'], standings['Position']))

        # default early-season rule
        home_pos.append(10 if played[home] < 3 else pos_lookup[home])
        away_pos.append(10 if played[away] < 3 else pos_lookup[away])

        # update table AFTER recording positions
        hg = row['FTHG']
        ag = row['FTAG']

        table.loc[home, 'GF'] += hg
        table.loc[home, 'GD'] += hg - ag

        table.loc[away, 'GF'] += ag
        table.loc[away, 'GD'] += ag - hg

        if hg > ag:
            table.loc[home, 'Pts'] += 3
        elif hg < ag:
            table.loc[away, 'Pts'] += 3
        else:
            table.loc[home, 'Pts'] += 1
            table.loc[away, 'Pts'] += 1

        played[home] += 1
        played[away] += 1

    df['HomeTablePos'] = home_pos
    df['AwayTablePos'] = away_pos
    df['TablePosDiff'] = df['AwayTablePos'] - df['HomeTablePos']

    return df

def add_rolling_features(team_df):
    ### Compute rolling averages ###
    rolling_cols = [
        'TablePosDiff',
        'GoalsFor',
        'Shots',
        'ShotsOnTarget',
        'Corners',
        'Fouls'
    ]

    for col in rolling_cols:

        team_df[f'{col}_Rolling5'] = (
            team_df
            .groupby(['Team', 'Venue'])[col]
            .transform(
                lambda x: x.shift(1)
                        .rolling(window=5, min_periods=1)
                        .mean()
            )
        )

    for col in rolling_cols:

        team_df[f'{col}_Rolling10'] = (
            team_df
            .groupby(['Team', 'Venue'])[col]
            .transform(
                lambda x: x.shift(1)
                        .rolling(window=10, min_periods=1)
                        .mean()
            )
        )

    for col in rolling_cols:

        team_df[f'{col}_Rolling5_all'] = (
            team_df
            .groupby(['Team'])[col]
            .transform(
                lambda x: x.shift(1)
                        .rolling(window=5, min_periods=1)
                        .mean()
            )
        )

    for col in rolling_cols:

        team_df[f'{col}_Rolling10_all'] = (
            team_df
            .groupby(['Team'])[col]
            .transform(
                lambda x: x.shift(1)
                        .rolling(window=10, min_periods=1)
                        .mean()
            )
        )
    return team_df

def build_match_dataset(team_df, df):
    
    # -------------------------
    # HOME FEATURES
    # -------------------------
    home_features = team_df[team_df['Venue'] == 'H'].copy()
    home_features = home_features.rename(columns={'Team': 'HomeTeam'})
    cols_to_prefix = [c for c in home_features.columns if c not in ['Date', 'Time', 'HomeTeam']]
    home_features = home_features.rename(columns={c: f'Home_{c}' for c in cols_to_prefix})

    # -------------------------
    # AWAY FEATURES
    # -------------------------
    away_features = team_df[team_df['Venue'] == 'A'].copy()
    away_features = away_features.rename(columns={'Team': 'AwayTeam'})
    cols_to_prefix = [c for c in away_features.columns if c not in ['Date', 'Time', 'AwayTeam']]
    away_features = away_features.rename(columns={c: f'Away_{c}' for c in cols_to_prefix})

    # -------------------------
    # BASE MATCH DATA (Explicitly keep ONLY core identification columns)
    # -------------------------
    core_cols = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HomeTablePos', 'AwayTablePos', 'TablePosDiff']
    matches = df[core_cols].copy()

    # -------------------------
    # MERGE HOME & AWAY (Include Time to prevent column duplication)
    # -------------------------
    matches = matches.merge(
        home_features,
        on=['Date', 'Time', 'HomeTeam'],
        how='left'
    )

    matches = matches.merge(
        away_features,
        on=['Date', 'Time', 'AwayTeam'],
        how='left'
    )

    # -------------------------
    # CLEAN
    # -------------------------
    matches = matches.sort_values(['Date', 'Time']).reset_index(drop=True)
    matches = matches.dropna(subset=['Home_GoalsFor', 'Away_GoalsFor'])

    return matches

def export_dataset(matches, path="dataset/24-25.csv"):
    # optional safety cleanup
    df = matches.copy()

    # sort for reproducibility
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    # remove rows that still have missing features (optional but common)
    df = df.dropna()

    # export
    df.to_csv(path, index=False)

    print(f"Dataset exported to {path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    raw_df = load_data("pl24-25.csv")
    df = add_table_positions(raw_df)   
    team_df = create_team_df(df)       
    team_df = add_rolling_features(team_df)
    matches = build_match_dataset(team_df, df)
    export_dataset(matches)