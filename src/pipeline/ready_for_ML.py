import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer


class FootballPreprocessor:

    def __init__(self, paths):

        # Allow either a string or a list of paths
        if isinstance(paths, str):
            paths = [paths]

        self.paths = paths

        self.encoder = LabelEncoder()
        self.preprocessor = None

    # --------------------------------------------------
    # LOAD ALL SEASONS
    # --------------------------------------------------

    def load_data(self):

        dfs = []

        for path in self.paths:

            season = path.split("/")[-1].replace(".csv", "")

            df = pd.read_csv(path)

            df["Date"] = pd.to_datetime(df["Date"])

            # Remember which season every match belongs to
            df["Season"] = season

            dfs.append(df)

        df = pd.concat(
            dfs,
            ignore_index=True
        )

        df = (
            df
            .sort_values(["Date", "Time"])
            .reset_index(drop=True)
        )

        return df

    # --------------------------------------------------
    # CREATE X AND y
    # --------------------------------------------------

    def prepare_features(self, df):

        y = df["FTR"]

        # Keep Season for walk-forward validation
        meta = df[
            [
                "Date",
                "Time",
                "Season"
            ]
        ].copy()

        drop_cols = [

            "Date",
            "Time",
            "Season",

            "FTR",
            "FTHG",
            "FTAG"

        ]

        X = df.drop(columns=drop_cols)

        return X, y, meta

    # --------------------------------------------------
    # FIT PREPROCESSOR
    # --------------------------------------------------

    def fit(self, X):

        categorical = [
            "HomeTeam",
            "AwayTeam"
        ]

        numerical = [
            c
            for c in X.columns
            if c not in categorical
        ]

        self.preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    StandardScaler(),
                    numerical
                ),

                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical
                )

            ]

        )

        self.preprocessor.fit(X)

        return self

    # --------------------------------------------------
    # TRANSFORM FEATURES
    # --------------------------------------------------

    def transform(self, X):

        return self.preprocessor.transform(X)

    # --------------------------------------------------
    # FIT + TRANSFORM
    # --------------------------------------------------

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    # --------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------

    def encode_target(self, y):

        return self.encoder.fit_transform(y)

    def transform_target(self, y):

        return self.encoder.transform(y)