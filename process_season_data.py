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
    
        # Result converted to points already exists
    # Use Points

    team_df['Win'] = team_df['Result'].map({
        'W': 1,
        'D': 0.5,
        'L': 0
    })


    for col in ['Points', 'Win']:

        team_df[f'{col}_Rolling5'] = (
            team_df
            .groupby(['Team','Venue'])[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(5, min_periods=1)
                .mean()
            )
        )

        team_df[f'{col}_Rolling10'] = (
            team_df
            .groupby(['Team','Venue'])[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(10, min_periods=1)
                .mean()
            )
        )


        team_df[f'{col}_Rolling5_all'] = (
            team_df
            .groupby('Team')[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(5, min_periods=1)
                .mean()
            )
        )


        team_df[f'{col}_Rolling10_all'] = (
            team_df
            .groupby('Team')[col]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(10, min_periods=1)
                .mean()
            )
        )
    return team_df

def build_match_dataset(team_df, df):

    # -------------------------
    # HOME FEATURES
    # -------------------------
    home_features = team_df[team_df['Venue'] == 'H'].copy()

    home_features = home_features.rename(
        columns={'Team': 'HomeTeam'}
    )

    # Keep only pre-match rolling features + table position diff
    home_keep = [
        'Date',
        'Time',
        'HomeTeam',

        'TablePosDiff',

        'TablePosDiff_Rolling5',
        'TablePosDiff_Rolling10',

        'GoalsFor_Rolling5',
        'GoalsFor_Rolling10',

        'Shots_Rolling5',
        'Shots_Rolling10',

        'ShotsOnTarget_Rolling5',
        'ShotsOnTarget_Rolling10',

        'Corners_Rolling5',
        'Corners_Rolling10',

        'Fouls_Rolling5',
        'Fouls_Rolling10',

        'TablePosDiff_Rolling5_all',
        'TablePosDiff_Rolling10_all',

        'GoalsFor_Rolling5_all',
        'GoalsFor_Rolling10_all',

        'Shots_Rolling5_all',
        'Shots_Rolling10_all',

        'ShotsOnTarget_Rolling5_all',
        'ShotsOnTarget_Rolling10_all',

        'Corners_Rolling5_all',
        'Corners_Rolling10_all',

        'Fouls_Rolling5_all',
        'Fouls_Rolling10_all',

        'Points_Rolling5',
        'Points_Rolling10',

        'Points_Rolling5_all',
        'Points_Rolling10_all',

        'Win_Rolling5',
        'Win_Rolling10',

        'Win_Rolling5_all',
        'Win_Rolling10_all'
    ]


    home_features = home_features[home_keep]


    # Prefix
    home_features = home_features.rename(
        columns={
            c: f"Home_{c}"
            for c in home_features.columns
            if c not in ['Date','Time','HomeTeam']
        }
    )


    # -------------------------
    # AWAY FEATURES
    # -------------------------
    away_features = team_df[team_df['Venue'] == 'A'].copy()

    away_features = away_features.rename(
        columns={'Team': 'AwayTeam'}
    )

    away_keep = [
        'Date',
        'Time',
        'AwayTeam',

        'TablePosDiff',

        'TablePosDiff_Rolling5',
        'TablePosDiff_Rolling10',

        'GoalsFor_Rolling5',
        'GoalsFor_Rolling10',

        'Shots_Rolling5',
        'Shots_Rolling10',

        'ShotsOnTarget_Rolling5',
        'ShotsOnTarget_Rolling10',

        'Corners_Rolling5',
        'Corners_Rolling10',

        'Fouls_Rolling5',
        'Fouls_Rolling10',

        'TablePosDiff_Rolling5_all',
        'TablePosDiff_Rolling10_all',

        'GoalsFor_Rolling5_all',
        'GoalsFor_Rolling10_all',

        'Shots_Rolling5_all',
        'Shots_Rolling10_all',

        'ShotsOnTarget_Rolling5_all',
        'ShotsOnTarget_Rolling10_all',

        'Corners_Rolling5_all',
        'Corners_Rolling10_all',

        'Fouls_Rolling5_all',
        'Fouls_Rolling10_all',

        'Points_Rolling5',
        'Points_Rolling10',

        'Points_Rolling5_all',
        'Points_Rolling10_all',

        'Win_Rolling5',
        'Win_Rolling10',

        'Win_Rolling5_all',
        'Win_Rolling10_all'
    ]

    away_features = away_features[away_keep]


    # -------------------------
    # BASE MATCH DATA
    # -------------------------
    matches = df[
        [
            'Date',
            'Time',
            'HomeTeam',
            'AwayTeam',
            'FTR',       # target
            'FTHG',      # keep temporarily for evaluation
            'FTAG'
        ]
    ].copy()


    # -------------------------
    # MERGE FEATURES
    # -------------------------
    matches = matches.merge(
        home_features,
        on=['Date','Time','HomeTeam'],
        how='left'
    )


    matches = matches.merge(
        away_features,
        on=['Date','Time','AwayTeam'],
        how='left'
    )


    matches = matches.sort_values(
        ['Date','Time']
    ).reset_index(drop=True)


    return matches

def export_dataset(matches, path="dataset/21-22.csv"):
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
    raw_df = load_data("pl21-22.csv")
    df = add_table_positions(raw_df)   
    team_df = create_team_df(df)       
    team_df = add_rolling_features(team_df)
    matches = build_match_dataset(team_df, df)
    export_dataset(matches)