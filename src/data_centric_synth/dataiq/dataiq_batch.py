from typing import Optional, Union

import numpy as np
import torch

from data_centric_synth.dataiq.metrics import METRICS, set_metrics_as_properties
from data_centric_synth.dataiq.numpy_helpers import (
    add_values_by_indices,
    add_values_sequentially,
    convert_to_numpy,
    get_ground_truth_probs,
    onehot2int,
)


class DataIQBatch:
    """Class for calculating the aleatoric uncertainty for models outputting
    probabilities over epochs.

    Attributes:
        n_samples (int): Number of samples (rows) in the dataset. Needs to be
            set at initialization to handle batching correctly.
        label_probs (np.ndarray): Probability of the ground truth label for
            each sample. np.ndarray of shape (n_samples, n_epochs)
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self._current_sample_idx = 0

        self._epoch_label_probs = np.zeros(n_samples)

        self._set_metrics_as_properties()

    @classmethod
    def _set_metrics_as_properties(cls) -> None:  # noqa
        """Set the metrics in dataiq.metrics.py as properties of the class.
        This allows them to be easily accessed using cls.metric_name."""
        set_metrics_as_properties(cls, label_probs_name="label_probs")

    def on_batch_end(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        indices: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> None:
        """Add the probabilities of the ground truth label for each sample.
        For each batch in an epoch, the probabilities are added to the
        _epoch_label_probs attribute. At the end of the epoch, the
        probabilities are added to the label_probs attribute. Note, samples must
        be input in the same order for each epoch (i.e. no shuffling), OR
        indices must be supplied.

        Args:
            y_true (np.ndarray): Ground truth labels. np.ndarray of shape
                (batch_size,)
            y_pred (np.ndarray): Predicted labels. np.ndarray of shape
                (batch_size, n_classes)
            indices (np.ndarray): Indices of the samples in the batch.
        """
        batch_size = y_true.shape[0]

        y_true, y_pred = convert_to_numpy(y_true), convert_to_numpy(y_pred)

        # convert labels to int if one-hot encoded
        y_true = onehot2int(array=y_true)

        # get the probabilities of the ground truth label
        ground_truth_probs = get_ground_truth_probs(y_true=y_true, y_pred=y_pred)

        # add the probabilities to the _epoch_label_probs attribute for the
        # current batch
        if indices is None:
            self._epoch_label_probs = add_values_sequentially(
                array=self._epoch_label_probs,
                values=ground_truth_probs,
                batch_size=batch_size,
                sample_index=self._current_sample_idx,
            )
        else:
            self._epoch_label_probs = add_values_by_indices(
                array=self._epoch_label_probs,
                values=ground_truth_probs,
                indices=indices,  # type: ignore
            )

        # update the current sample index
        self._current_sample_idx += batch_size
        # if the current sample index is equal to the number of samples, then
        # the epoch is over and the probabilities for the epoch are added to
        # the probs attribute
        if self._current_sample_idx == self.n_samples:
            self._add_epoch_probs()
            self._current_sample_idx = 0

    def _add_epoch_probs(self) -> None:
        """Add the probabilities for the current epoch to the probs attribute."""
        self.label_probs: np.ndarray = (
            np.hstack(
                (
                    self.label_probs,
                    self._epoch_label_probs[:, None],
                ),
            )
            if hasattr(self, "label_probs")
            else self._epoch_label_probs[:, None]
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
