"""This module defines the following routines used by the 'split' step of the regression recipe.

- ``process_splits``: Defines customizable logic for processing & cleaning the training, validation,
  and test datasets produced by the data splitting procedure.
"""

import pandas as pd


def process_splits(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Perform additional processing on the split datasets.

    Args:
        train_df (pd.DataFrame): The training dataset produced by the data splitting procedure.
        validation_df (pd.DataFrame): The validation dataset produced by the data splitting procedure.
        test_df (pd.DataFrame): The test dataset produced by the data splitting procedure.

    Returns:
        tuple: Tuple of pandas DataFrames of processed splits
    """

    def process(df: pd.DataFrame):
        # Drop invalid data points
        cleaned = df.dropna()
        # Filter out invalid fare amounts and trip distance
        cleaned = cleaned[
            (cleaned["fare_amount"] > 0)
            & (cleaned["trip_distance"] < 400)
            & (cleaned["trip_distance"] > 0)
            & (cleaned["fare_amount"] < 1000)
        ]

        return cleaned

    return process(train_df), process(validation_df), process(test_df)
