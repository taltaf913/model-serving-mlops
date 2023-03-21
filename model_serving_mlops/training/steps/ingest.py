"""This module defines the following routines used by the 'ingest' step of the regression recipe.

- ``load_file_as_dataframe``: Defines customizable logic for parsing dataset formats that are not
  natively parsed by MLflow Recipes (i.e. formats other than Parquet, Delta, and Spark SQL).
"""

import logging

import pandas as pd

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> pd.DataFrame:
    """Load content from the specified dataset file as a Pandas DataFrame.

    This method is used to load dataset types that are not natively managed by MLflow Recipes
    (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is
    called once for each file in the dataset, and MLflow Recipes automatically combines the
    resulting DataFrames together.

    Args:
        file_path (str): The path to the dataset file.
        file_format (str): The file format string, such as "csv".

    Raises:
        NotImplementedError: Unsupported file format.

    Returns:
        pd.DataFrame: A Pandas DataFrame representing the content of the specified file.
    """
    if file_format == "csv":
        _logger.warning(
            "Loading dataset CSV using `pandas.read_csv()` with default arguments and assumed index"
            " column 0 which may not produce the desired schema. If the schema is not correct, you"
            " can adjust it by modifying the `load_file_as_dataframe()` function in"
            " `steps/ingest.py`"
        )
        return pd.read_csv(file_path, index_col=0)
    else:
        raise NotImplementedError
