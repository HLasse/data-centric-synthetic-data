"""Code to sculpt with DataIQ."""

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from xgboost import XGBClassifier

from data_centric_synth.data_models.data_sculpting import (
    DataSegments,
    ProcessedData,
    SculptedData,
    StratifiedIndices,
)
from data_centric_synth.dataiq.dataiq_batch import DataIQBatch
from data_centric_synth.dataiq.dataiq_gradient_boosting import DataIQGradientBoosting
from data_centric_synth.experiments.models import IMPLEMENTED_DATA_CENTRIC_METHODS


def add_target_col(X: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    df = X.copy()
    df["target"] = target
    return df


def get_datacentric_segments_from_sculpted_data(
    sculpted_data: SculptedData,
    subsets: List[Literal["easy", "ambi", "hard"]],
) -> DataSegments:
    """Get the number of samples in each subset of the sculpted data.
    The number of samples is set to 0 if the subset is not in `subsets`.

    Args:
        sculpted_data (SculptedData): The sculpted data.
        subsets (List[Literal["easy", "ambi", "hard"]]): The subsets of the
            sculpted data to get the number of samples for.

    Returns:
        DataIQSegments: The number of samples in each subset of the sculpted data.
    """
    n_easy = sculpted_data.indices.easy.size if "easy" in subsets else 0
    if sculpted_data.indices.ambiguous is None:
        n_ambi = 0
    else:
        n_ambi = sculpted_data.indices.ambiguous.size if "ambi" in subsets else 0
    n_hard = sculpted_data.indices.hard.size if "hard" in subsets else 0
    return DataSegments(
        n_easy=n_easy,
        n_ambiguous=n_ambi,
        n_hard=n_hard,
    )


def extract_subsets_from_sculpted_data(
    data: SculptedData,
    subsets: List[Literal["easy", "ambi", "hard"]],
) -> ProcessedData:
    """Extract subsets of the sculpted data from the `SculptedData` object."""
    # If the only subsets is the ambiguous subset and there are less than 10
    # samples in the ambiguous subset, then return an empty dataset.
    if (
        len(subsets) == 1
        and "ambi" in subsets
        and (data.y_ambiguous is None or len(data.y_ambiguous) < 10)
    ):
        return ProcessedData(
            dataset=pd.DataFrame(),
            data_segments=DataSegments(n_easy=0, n_ambiguous=0, n_hard=0),
            statistical_likeness=None,
            detection_auc=None,
        )

    out: List[pd.DataFrame] = []
    if "easy" in subsets:
        out.append(add_target_col(X=data.X_easy, target=data.y_easy))
    if (
        "ambi" in subsets
        and data.X_ambiguous is not None
        and data.y_ambiguous is not None
        and len(data.y_ambiguous)
        >= 10  # require a minimum amount of samples for the ambiguous subset
    ):
        out.append(add_target_col(X=data.X_ambiguous, target=data.y_ambiguous))
    if "hard" in subsets:
        out.append(add_target_col(X=data.X_hard, target=data.y_hard))
    return ProcessedData(
        dataset=pd.concat(out),
        data_segments=get_datacentric_segments_from_sculpted_data(
            sculpted_data=data,
            subsets=subsets,
        ),
        statistical_likeness=None,  # setting to none for the postprocessing
        detection_auc=None,
    )


def stratify_samples(
    label_probabilities: np.ndarray,
    data_uncertainty: np.ndarray,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
) -> StratifiedIndices:
    """Stratify samples into easy, ambigious, and hard examples based on the
    confidence and aleatoric uncertainty of the model/data. The percentile
    threshold determines how many samples are in each category.

    Args:
        label_probabilities (np.ndarray): Mean probabilities of the ground truth
            label for each sample.
        data_uncertainty (np.ndarray): Aleatoric uncertainty epistemic variability
            of the data for each sample.
        percentile_threshold (int): Percentile threshold to use for stratifying
            the samples. If `percentile_threshold` is not None, then the
            `data_centric_threshold` is ignored.
        data_centric_threshold (float): Data centric threshold to use for
            stratifying the samples. If `data_centric_threshold` is not None, then
            the `percentile_threshold` is ignored.

    Returns:
        StratifiedIndices: Indices of the stratified samples
    """
    if percentile_threshold is not None and data_centric_threshold is not None:
        raise ValueError(
            "Must provide only one of percentile_threshold or data_centric_threshold",
        )

    if percentile_threshold is not None:
        data_uncertainty_below_threshold = data_uncertainty <= np.percentile(
            data_uncertainty,
            percentile_threshold,
        )
    elif data_centric_threshold is not None:
        data_uncertainty_below_threshold = data_uncertainty <= data_centric_threshold
    else:
        raise ValueError(
            "Must provide either percentile_threshold or data_centric_threshold",
        )
    confidence_treshold_width = 0.5
    confidence_threshold_lower = 0.5 - confidence_treshold_width / 2
    confidence_threshold_upper = 0.5 + confidence_treshold_width / 2

    hard_idx = np.where(
        (label_probabilities <= confidence_threshold_lower)
        & (data_uncertainty_below_threshold),
    )[0]
    easy_idx = np.where(
        (label_probabilities >= confidence_threshold_upper)
        & (data_uncertainty_below_threshold),
    )[0]

    # ambigious ids are those not in hard_idx or easy_idx
    all_ids = np.arange(len(label_probabilities))
    ambigious_idx = np.setdiff1d(all_ids, np.concatenate((hard_idx, easy_idx)))
    return StratifiedIndices(easy=easy_idx, ambiguous=ambigious_idx, hard=hard_idx)


def get_sculpted_data(
    X: pd.DataFrame,
    y: pd.Series,
    stratified_indices: StratifiedIndices,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
) -> SculptedData:
    """Get the stratified samples from the dataset."""
    X_easy = X.loc[stratified_indices.easy]
    y_easy = y[stratified_indices.easy]
    if stratified_indices.ambiguous is not None:
        X_ambigious = X.loc[stratified_indices.ambiguous]
        y_ambigious = y[stratified_indices.ambiguous]
    else:
        X_ambigious = None
        y_ambigious = None
    X_hard = X.loc[stratified_indices.hard]
    y_hard = y[stratified_indices.hard]
    return SculptedData(
        X_easy=X_easy,
        X_ambiguous=X_ambigious,
        X_hard=X_hard,
        y_easy=y_easy,
        y_ambiguous=y_ambigious,
        y_hard=y_hard,
        indices=stratified_indices,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )


# Rewrite to work with any method (dataiq, datamaps, cleanlab)
# def sculpt_by_thresholds(
#     X: pd.DataFrame,
#     y: pd.Series,
#     dataiq: Union[DataIQGradientBoosting, DataIQBatch],
#     percentile_thresholds: list[int],
# ) -> list[SculptedData]:
#     """Sculpt the dataset using DataIQ."""
#     return [
#         stratify_and_sculpt_with_dataiq(
#             X=X, y=y, dataiq=dataiq, percentile_threshold=pt
#         )
#         for pt in percentile_thresholds
#     ]


def stratify_and_sculpt_with_dataiq(
    X: pd.DataFrame,
    y: pd.Series,
    dataiq: Union[DataIQGradientBoosting, DataIQBatch],
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    method: Literal["dataiq", "datamaps"],
) -> SculptedData:
    """Stratify the dataset into easy, ambiguous and hard examples
    based on the confidence and aleatoric uncertainty

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        dataiq (Union[DataIQGradientBoosting, DataIQBatch]): DataIQ object
        percentile_threshold (int): Percentile threshold for aleatoric uncertainty
            to use for stratifying
        data_centric_threshold (float): Data centric threshold for aleatoric
            or epistemic uncertainty to use for stratifying
        method (Literal["dataiq", "datamaps"]): Method to use for data centric
            uncertainty

    Returns:
        SculptedData: Stratified dataset
    """
    if method == "dataiq":
        data_uncertainty = dataiq.aleatoric_uncertainty  # type: ignore
    elif method == "datamaps":
        data_uncertainty = dataiq.epistemic_variability  # type: ignore
    else:
        raise ValueError(f"Unknown data centric method {method}")

    stratified_indices = stratify_samples(
        label_probabilities=dataiq.average_confidence,  # type: ignore
        data_uncertainty=data_uncertainty,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )
    return get_sculpted_data(
        X=X,
        y=y,
        stratified_indices=stratified_indices,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
    )


def sculpt_with_dataiq(
    X: pd.DataFrame,
    y: pd.Series,
    percentile_threshold: Optional[int],
    data_centric_threshold: Optional[float],
    method: Literal["dataiq", "datamaps"],
) -> SculptedData:
    """Sculpt the given data with DataIQ. Fits an XGBoost model and uses the model
    with DataIQ to stratify the data into easy, ambiguous, and hard examples.

    Args:
        X (pd.DataFrame): The training features.
        y (pd.Series): The training labels.
        percentile_threshold (int): The percentile threshold for aleatoric uncertainty
            used for making the easy/ambiguous/hard split.
        data_centric_threshold (float): The data centric threshold for making the
            easy/ambiguous/hard split.
        method (Literal["dataiq", "datamaps"]): The method to use for sculpting.

        Returns:
            SculptedData: The sculpted data.
    """
    dataiq = fit_dataiq(X=X, y=y)
    return stratify_and_sculpt_with_dataiq(
        X=X,
        y=y,
        dataiq=dataiq,
        percentile_threshold=percentile_threshold,
        data_centric_threshold=data_centric_threshold,
        method=method,
    )


def fit_dataiq(X: pd.DataFrame, y: pd.Series) -> DataIQGradientBoosting:
    """Fit a DataIQ model on the given data.

    Args:
        X (pd.DataFrame): The training features.
        y (pd.Series): The training labels.

    Returns:
        DataIQGradientBoosting: The fitted DataIQ model."""
    model = XGBClassifier()
    model.fit(X, y)
    dataiq = DataIQGradientBoosting()
    dataiq.evaluate_gradient_boosting(
        model=model,
        X=X,
        y=y,
    )
    return dataiq


def sculpt_with_cleanlab(
    X: pd.DataFrame,
    y: pd.Series,
    data_centric_threshold: Optional[float],
    model = XGBClassifier(),
) -> SculptedData:
    """Sculpt the given data with Cleanlab. Uses the Cleanlab model to stratify the
    data into easy and hard examples. Note that cleanlab does not have a notion of
    ambiguous examples.

    Args:
        X (pd.DataFrame): The training features.
        y (pd.Series): The training labels.
        data_centric_threshold (float): The data centric threshold for making the
            easy/ambiguous/hard split.
    """

    label_issues = get_cleanlab_label_issue_df(X=X, y=y, model=model)
    if data_centric_threshold is None:
        easy_indices = np.where(~label_issues["is_label_issue"])[0]
        hard_indices = np.where(label_issues["is_label_issue"])[0]
    else:
        easy_indices = np.where(label_issues["label_quality"] > data_centric_threshold)[
            0
        ]
        hard_indices = np.where(
            label_issues["label_quality"] <= data_centric_threshold,
        )[0]

    return get_sculpted_data(
        X=X,
        y=y,
        stratified_indices=StratifiedIndices(
            easy=easy_indices,
            ambiguous=None,
            hard=hard_indices,
        ),
        percentile_threshold=None,
        data_centric_threshold=data_centric_threshold,
    )


def get_cleanlab_label_issue_df(X: pd.DataFrame, y: pd.Series, model = XGBClassifier()) -> pd.DataFrame:
    cl = CleanLearning(model)
    _ = cl.fit(X, y)
    label_issues = cl.get_label_issues()
    if label_issues is None:
        raise ValueError("Model not fit?")
    return label_issues




def sculpt_data_by_method(
    X: pd.DataFrame,
    y: pd.Series,
    data_centric_threshold: Optional[float],
    percentile_threshold: Optional[int],
    data_centric_method: IMPLEMENTED_DATA_CENTRIC_METHODS,
) -> SculptedData:
    if data_centric_method == "cleanlab":
        sculpted_data = sculpt_with_cleanlab(
            X=X,
            y=y,
            data_centric_threshold=data_centric_threshold,
        )
    elif data_centric_method in ["dataiq", "datamaps"]:
        sculpted_data = sculpt_with_dataiq(
            X=X,
            y=y,
            percentile_threshold=percentile_threshold,
            data_centric_threshold=data_centric_threshold,
            method=data_centric_method,
        )
    else:
        raise ValueError(
            f"Unknown data centric method: {data_centric_method}. Should be"
            + " one of ['dataiq', 'datamaps', 'cleanlab'].",
        )

    return sculpted_data
