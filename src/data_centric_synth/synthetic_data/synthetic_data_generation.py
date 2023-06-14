from typing import Any, Dict, List, Literal

import pandas as pd
from synthcity.metrics.eval_detection import SyntheticDetectionMLP
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from data_centric_synth.data_models.data_sculpting import (
    AlphaPrecisionMetrics,
    ProcessedData,
    SculptedData,
)
from data_centric_synth.data_sculpting.datacentric_sculpting import (
    add_target_col,
    get_datacentric_segments_from_sculpted_data,
)
from data_centric_synth.experiments.statistical_fidelity import StatisticalFidelity


def fit_and_generate_synth_data(
    data: pd.DataFrame,
    model_name: str,
    model_params: Dict[str, Any],
    target_column: str = "target",
) -> pd.DataFrame:
    """Fit a synthetic data generator from synthcity and generate synthetic data
    of the same length.

    Args:
        data (pd.DataFrame): The data to fit the synthetic data generator on.
        model_name (str): The name of the synthetic data generator to use.
        model_params (Dict[str, Any]): The hyperparameters of the synthetic data generator.
        target_column (str, optional): The name of the target column. Defaults to "target".

    Returns:
        pd.DataFrame: The generated synthetic data. Length is equal to the length of the input data.
    """
    loader = GenericDataLoader(data=data, target_column=target_column)
    syn_model = Plugins().get(model_name, **model_params)
    syn_model.fit(loader)
    return syn_model.generate(count=len(data)).dataframe()


def generate_synth_data_from_sculpted_data(
    data: SculptedData,
    combine: List[Literal["easy", "ambi", "hard"]],
    synthetic_model: str,
    synthetic_model_params: Dict[str, Any],
) -> ProcessedData:
    """Generate synthetic data from the sculpted data. Fits separate synthetic data
    generators on the different sculpted data subsets provided in `combine` and
    combines them in the end.

    Args:
        data (SculptedData): The sculpted data.
        combine (List[Literal["easy", "ambi", "hard"]]): The subsets of the sculpted data to combine.
        synthetic_model (str): The name of the synthetic data generator to use.
        synthetic_model_params (Dict[str, Any]): The hyperparameters of the synthetic data generator.

    Returns:
        pd.DataFrame: The generated synthetic data.
    """
    synth_data = pd.DataFrame()
    if "easy" in combine:
        easy_data = add_target_col(X=data.X_easy, target=data.y_easy)
        synth_data = pd.concat(
            [
                synth_data,
                fit_and_generate_synth_data(
                    data=easy_data,
                    model_name=synthetic_model,
                    model_params=synthetic_model_params,
                ),
            ],
        )

    if "ambi" in combine:
        ambigious_data = add_target_col(X=data.X_ambiguous, target=data.y_ambiguous)  # type: ignore
        synth_data = pd.concat(
            [
                synth_data,
                fit_and_generate_synth_data(
                    data=ambigious_data,
                    model_name=synthetic_model,
                    model_params=synthetic_model_params,
                ),
            ],
        )
    if "hard" in combine:
        hard_data = add_target_col(X=data.X_hard, target=data.y_hard)
        # only append if there are hard samples
        if not hard_data.empty:
            synth_data = pd.concat(
                [
                    synth_data,
                    fit_and_generate_synth_data(
                        data=hard_data,
                        model_name=synthetic_model,
                        model_params=synthetic_model_params,
                    ),
                ],
            )
    synth_data = synth_data.reset_index(drop=True)

    return ProcessedData(
        dataset=synth_data,
        data_segments=get_datacentric_segments_from_sculpted_data(
            sculpted_data=data,
            subsets=combine,
        ),
        statistical_likeness=None,
        detection_auc=None,
    )


def concat_sculpted_data(
    sculpted_data: SculptedData,
    subsets: List[str],
) -> pd.DataFrame:
    org_data: List[pd.DataFrame] = []
    if "easy" in subsets:
        org_data.append(
            add_target_col(X=sculpted_data.X_easy, target=sculpted_data.y_easy),
        )
    if "ambi" in subsets:
        org_data.append(
            add_target_col(
                X=sculpted_data.X_ambiguous,
                target=sculpted_data.y_ambiguous,  # type: ignore
            ),
        )
    if "hard" in subsets:
        org_data.append(
            add_target_col(X=sculpted_data.X_hard, target=sculpted_data.y_hard),
        )

    return pd.concat(org_data).reset_index(drop=True)


def calculate_detection_auc_from_sculpted_data(
    sculpted_data: SculptedData,
    synthetic_data: pd.DataFrame,
    subsets: List[str],
) -> float:
    """Calculate the detection AUC for synthetic data across different
    subsets of the sculpted data. Combines the subsets defined in subsets and
    calculates the detection AUC for the combined data."""
    org_data = concat_sculpted_data(sculpted_data, subsets)
    return evaluate_detection_auc(
        X=org_data,
        X_syn=synthetic_data,
    )


def evaluate_detection_auc(X: pd.DataFrame, X_syn: pd.DataFrame) -> float:
    """Calculate the detection AUC for synthetic data using a MLP."""
    return SyntheticDetectionMLP().evaluate(
        X_gt=GenericDataLoader(data=X, target_column="target"),
        X_syn=GenericDataLoader(data=X_syn, target_column="target"),
    )["mean"]


def calculate_alpha_precision_from_sculpted_data(
    sculpted_data: SculptedData,
    synthetic_data: pd.DataFrame,
    subsets: List[str],
) -> AlphaPrecisionMetrics:
    """Calculate the alpha precision metrics for synthetic data across different
    subsets of the sculpted data. Combines the subsets defined in subsets and
    calculates the alpha precision metrics for the combined data."""
    org_data = concat_sculpted_data(sculpted_data, subsets)
    return StatisticalFidelity().calculate_alpha_precision(
        X=org_data, X_syn=synthetic_data
    )
