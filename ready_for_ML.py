import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FootballPreprocessor:

    def __init__(self, path):

        self.path = path

        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.preprocessor = None


    def load_data(self):

        df = pd.read_csv(self.path)

        df['Date'] = pd.to_datetime(df['Date'])

        # Ensure chronological order
        df = (
            df
            .sort_values(['Date', 'Time'])
            .reset_index(drop=True)
        )

        return df


    def prepare_features(self, df):

        # -----------------------
        # TARGET
        # -----------------------

        y = df['FTR']


        # -----------------------
        # REMOVE UNAVAILABLE DATA
        # -----------------------

        drop_cols = [
            'Date',
            'Time',

            'FTR',
            'FTHG',
            'FTAG'
        ]


        X = df.drop(
            columns=drop_cols
        )


        return X, y



    def split_data(
        self,
        X,
        y,
        train_ratio=0.8
    ):

        split = int(
            len(X) * train_ratio
        )


        X_train = X.iloc[:split]
        X_test = X.iloc[split:]


        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        return (
            X_train,
            X_test,
            y_train,
            y_test
        )



    def preprocess(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):


        # -----------------------
        # Encode target
        # -----------------------

        y_train = self.encoder.fit_transform(
            y_train
        )

        y_test = self.encoder.transform(
            y_test
        )


        # -----------------------
        # Identify columns
        # -----------------------

        categorical = [
            'HomeTeam',
            'AwayTeam'
        ]


        numerical = [
            c for c in X_train.columns
            if c not in categorical
        ]


        # -----------------------
        # Transform pipeline
        # -----------------------

        self.preprocessor = ColumnTransformer(

            transformers=[

                (
                    'num',
                    StandardScaler(),
                    numerical
                ),

                (
                    'cat',
                    OneHotEncoder(
                        handle_unknown='ignore'
                    ),
                    categorical
                )

            ]

        )


        X_train = self.preprocessor.fit_transform(
            X_train
        )


        X_test = self.preprocessor.transform(
            X_test
        )


        return (
            X_train,
            X_test,
            y_train,
            y_test
        )



    def run(self):

        df = self.load_data()


        X, y = self.prepare_features(
            df
        )


        X_train, X_test, y_train, y_test = self.split_data(
            X,
            y
        )


        return self.preprocess(
            X_train,
            X_test,
            y_train,
            y_test
        )