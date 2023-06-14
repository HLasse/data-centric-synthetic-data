from typing import Optional, Union

import numpy as np
import torch


def onehot2int(array: np.ndarray) -> np.ndarray:
    """Convert (possibly) one-hot encoded labels to integers. If the array
    is not one-hot encoded, then the array is returned.

    Args:
        array (np.ndarray): Array to convert. np.ndarray of shape
            (batch_size, n_classes) or (batch_size,)

    Returns:
        np.ndarray: Array with one-hot encoded labels converted to integers."""
    if len(array.shape) > 1:
        array = np.argmax(array, axis=1)
    return array


def get_ground_truth_probs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Get the probabilities of the ground truth label for each sample in
    the batch.

    Args:
        y_true (np.ndarray): Ground truth labels. np.ndarray of shape
            (batch_size,)
        y_pred (np.ndarray): Predicted labels. np.ndarray of shape
            (batch_size, n_classes)

    Returns:
        np.ndarray: Probabilities of the ground truth label for each sample.
        np.ndarray of shape (batch_size,)
    """
    return y_pred[np.arange(y_pred.shape[0]), y_true]


def convert_to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a torch tensor to a numpy array.

    Args:
        x (Union[np.ndarray, torch.Tensor]): Array to be converted.

    Returns:
        np.ndarray: Converted array.
    """
    return x.detach().numpy() if isinstance(x, torch.Tensor) else x


def add_values_to_array(
    array: np.ndarray,
    values: np.ndarray,
    indices: Optional[np.ndarray],
    sample_index: int,
    batch_size: int,
) -> np.ndarray:
    if indices is None:
        return add_values_sequentially(
            array=array,
            values=values,
            batch_size=batch_size,
            sample_index=sample_index,
        )
    return add_values_by_indices(array=array, values=values, indices=indices)


def add_values_sequentially(
    array: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    sample_index: int,
) -> np.ndarray:
    array[sample_index : sample_index + batch_size] = values
    return array


def add_values_by_indices(
    array: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    array[indices] = values
    return array
