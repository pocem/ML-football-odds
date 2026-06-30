import pandas as pd


class WalkForwardValidator:

    def __init__(self, seasons):

        self.seasons = seasons


    def split(self, meta):

        unique_seasons = sorted(
            meta["Season"].unique()
        )

        for i in range(1, len(unique_seasons)):

            train_seasons = unique_seasons[:i]
            test_season = unique_seasons[i]


            train_idx = meta[
                meta["Season"].isin(train_seasons)
            ].index


            test_idx = meta[
                meta["Season"] == test_season
            ].index


            yield (
                train_idx,
                test_idx,
                train_seasons,
                test_season
            )