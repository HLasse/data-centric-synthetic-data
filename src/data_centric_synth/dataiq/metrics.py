"""Metrics related to data uncertainty"""
import numpy as np


def average_confidence(label_probs: np.ndarray) -> np.ndarray:
    """Calculate the average confidence of the model for each sample.

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: The average confidence of the model for each sample.
            np.ndarray of shape (n_samples,)
    """
    return np.mean(label_probs, axis=1)


def aleatoric_uncertainty(label_probs: np.ndarray) -> np.ndarray:
    """Compute the aleatoric uncertainty of the ground truth label
    probability across epochs

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: Aleatoric uncertainty. np.ndarray of shape (n_samples,)
    """
    preds = label_probs
    return np.mean(preds * (1 - preds), axis=1)


def epistemic_variability(label_probs: np.ndarray) -> np.ndarray:
    """Compute the epistemic variability of the ground truth label
    probability across epochs. This is the datamaps aproach. See
    https://arxiv.org/abs/1906.02276

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: Epistemic variability. np.ndarray of shape (n_samples,)
    """
    preds = label_probs
    return np.std(preds, axis=1)


def correctness(label_probs: np.ndarray) -> np.ndarray:
    """Compute the proportion of times a sample is predicted correctly across
    epochs. Only defined for binary classification.

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: Correctness. np.ndarray of shape (n_samples,)
    """
    return np.mean(label_probs > 0.5, axis=1)


def entropy(label_probs: np.ndarray) -> np.ndarray:
    """Compute the predictive entropy of the ground truth label probability
    across epochs

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: Entropy. np.ndarray of shape (n_samples,)
    """
    return -1 * np.sum(label_probs * np.log(label_probs + 1e-12), axis=-1)


def mutual_information(label_probs: np.ndarray) -> np.ndarray:
    """Compute the mutual information of the ground truth label probability
    across epochs

    Args:
        label_probs (np.ndarray): The probabilities of the ground truth label
            for each sample. np.ndarray of shape (n_samples, n_epochs)
    Returns:
        np.ndarray: Mutual information. np.ndarray of shape (n_samples,)
    """
    ent = entropy(label_probs)
    average_conf = average_confidence(label_probs)
    average_conf_entropy = entropy(average_conf)

    return ent - average_conf_entropy


METRICS = {
    "average_confidence": average_confidence,
    "aleatoric_uncertainty": aleatoric_uncertainty,
    "epistemic_variability": epistemic_variability,
    "correctness": correctness,
    "entropy": entropy,
    "mutual_information": mutual_information,
}


def set_metrics_as_properties(cls, label_probs_name: str):  # noqa
    """Set all metrics as properties of the class"""
    for metric_name, metric_func in METRICS.items():
        setattr(
            cls,
            metric_name,
            property(
                lambda self, metric=metric_func: metric(getattr(self, label_probs_name)),  # type: ignore
            ),
        )
