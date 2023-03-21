"""This module defines the following routines used by the 'train' step of the regression recipe.

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""

from sklearn.linear_model import SGDRegressor


def estimator_fn() -> SGDRegressor:
    """Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
      The estimator's input and output signatures should be compatible with scikit-learn
      estimators.

    Returns:
        SGDRegressor: *unfitted* SKLearn SGDRegressor estimator
    """
    return SGDRegressor(random_state=42)
