from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier  # type: ignore
from xgboost import XGBClassifier

from data_centric_synth.dataiq.metrics import METRICS, set_metrics_as_properties
from data_centric_synth.dataiq.numpy_helpers import get_ground_truth_probs, onehot2int


class DataIQGradientBoosting:
    """Class for calculating the aleatoric uncertainty for gradoient boosting
    models (XGBoost and CatBoost supported)
    """

    def __init__(self):
        self._set_metrics_as_properties()

    @classmethod
    def _set_metrics_as_properties(cls) -> None:  # noqa
        """Set the metrics in dataiq.metrics.py as properties of the class.
        This allows them to be easily accessed using cls.metric_name."""
        set_metrics_as_properties(cls, label_probs_name="label_probs")

    def evaluate_gradient_boosting(
        self,
        model: Union[XGBClassifier, CatBoostClassifier],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        n_iterations: int = 10,
    ) -> None:
        """
        Calculates aleatoric uncertainty estimates for gradient boosting methods
        using either XGBoost or CatBoost. The class probabilities are gathered
        by looping through the trees built in each iteration up to a maximum
        of n_iterations.
        """
        if isinstance(model, XGBClassifier):
            self._evaluate_xgboost(model=model, X=X, y=y, n_iterations=n_iterations)
        elif isinstance(model, CatBoostClassifier):
            self._evaluate_catboost(model=model, X=X, y=y, n_iterations=n_iterations)

    def _evaluate_xgboost(
        self,
        model: XGBClassifier,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_iterations: int,
    ) -> None:
        """
        Sets aleatoric uncertainty estimates for XGBoost classifiers.
        """
        n_iter = self._get_number_of_iterations(
            best_iteration=model.best_iteration,
            max_iterations=n_iterations,
        )
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        y = onehot2int(y)  # type: ignore

        for i in np.linspace(0, model.best_iteration + 1, num=n_iter):
            i = int(i)  # noqa
            probabilities = model.predict_proba(X, iteration_range=(0, i))
            ground_truth_probs = get_ground_truth_probs(y_true=y, y_pred=probabilities)
            self._add_epoch_probs(label_probs=ground_truth_probs)

    def _evaluate_catboost(
        self,
        model: CatBoostClassifier,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_iterations: int,
    ) -> None:
        """
        Sets aleatoric uncertainty estimates for Catboost classifiers"""
        n_model_iters = model.get_all_params()["iterations"]
        n_iter = self._get_number_of_iterations(
            best_iteration=n_model_iters,
            max_iterations=n_iterations,
        )
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        y = onehot2int(y)  # type: ignore

        for i in np.linspace(0, n_model_iters, num=n_iter):
            i = int(i)  # noqa
            probabilities = model.predict_proba(X=X, ntree_start=0, ntree_end=i)
            ground_truth_probs = get_ground_truth_probs(y_true=y, y_pred=probabilities)
            self._add_epoch_probs(label_probs=ground_truth_probs)

    def _add_epoch_probs(self, label_probs: np.ndarray) -> None:
        """Add the probabilities for the current epoch to the probs attribute."""
        self.label_probs: np.ndarray = (
            np.hstack(
                (
                    self.label_probs,
                    label_probs[:, None],
                ),
            )
            if hasattr(self, "label_probs")
            else label_probs[:, None]
        )

    def get_all_metrics(self) -> dict:
        """Calculate all metrics and return them in a dictionary.

        Returns:
            dict: Dictionary of metrics.
        """
        if self.label_probs is None:
            raise ValueError("No label probabilities have been added.")
        return {
            metric_name: metric_fun(self.label_probs)
            for metric_name, metric_fun in METRICS.items()
        }

    @staticmethod
    def _get_number_of_iterations(best_iteration: int, max_iterations: int) -> int:
        """Gets the maximum number of iterations for predict proba"""
        return min(best_iteration, max_iterations)
