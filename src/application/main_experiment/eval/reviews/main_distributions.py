"""Analysis of the distribution of scores across the tasks by generative model, classifier, and dataset."""


from typing import List

import pandas as pd
import plotnine as pn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from application.constants import DATA_DIR, RESULTS_DIR
from application.main_experiment.eval.summary_table import (
    aggregate_performance_across_tasks,
    aggregate_performance_across_tasks_no_merging,
    rename_easy_ambi_to_no_hard,
    rename_easy_ambiguous_hard_to_easy_ambi_hard,
)
from data_centric_synth.evaluation.extraction import get_experiment3_suite
from data_centric_synth.experiments.models import (
    IMPLEMENTED_DATA_CENTRIC_METHODS,
    get_default_synthetic_model_suite,
)
from statsmodels.regression.linear_model import RegressionResultsWrapper

from data_centric_synth.evaluation.summary_helpers import get_performance_dfs


def plot_density(df: pd.DataFrame) -> pn.ggplot:
    return (
        pn.ggplot(df, pn.aes(x="mean", color="pre_post", fill="pre_post"))
        + pn.geom_density(alpha=0.3)
        + pn.facet_grid("synthetic_model_type ~ dataset_id", scales="free_y")
        + pn.theme_minimal()
        + pn.theme(legend_position="bottom")
    )


def fit_lm(df: pd.DataFrame, formula: str) -> RegressionResultsWrapper:
    md = smf.ols(
        formula,
        df,
    )
    return md.fit()


def extract_estimate_and_p_value_from_lm(
    model: RegressionResultsWrapper,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimate": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
        }
    ).reset_index()


def get_processing_estimates(params_df: pd.DataFrame) -> pd.DataFrame:
    return params_df.query(
        "index in ['preprocessing_strategy[T.easy_hard]', 'postprocessing_strategy[T.no_hard]']"
    )


def fit_lm_and_get_processing_estimates(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    mdf = fit_lm(df=df, formula=formula)
    results = extract_estimate_and_p_value_from_lm(model=mdf)
    return get_processing_estimates(params_df=results)


if __name__ == "__main__":
    data_centric_methods: List[IMPLEMENTED_DATA_CENTRIC_METHODS] = [
        "cleanlab",
        "dataiq",
        "datamaps",
    ]
    synthetic_models = [*get_default_synthetic_model_suite(), "None"]

    # dataset_dirs = RESULTS_DIR / "experiment3" / "data"
    dataset_dirs = DATA_DIR / "main_experiment" / "data"
    plot_save_dir = (
        RESULTS_DIR / "main_experiment" / "plots" / "appendix" / "distributions"
    )
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    experiment_suite = get_experiment3_suite(dataset_dirs)

    performance_dfs = get_performance_dfs(
        data_centric_methods, synthetic_models, experiment_suite
    )
    performances_dfs = rename_easy_ambi_to_no_hard(performance_dfs)
    performance_dfs = rename_easy_ambiguous_hard_to_easy_ambi_hard(performance_dfs)

    data_centric_method = "cleanlab"

    # no aggregation happening here as we supply all the aggregation columns
    # in by_cols
    combined_unnormalized = (
        aggregate_performance_across_tasks_no_merging(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == @data_centric_method & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            by_cols=[
                "synthetic_model_type",
                "dataset_id",
                "random_seed",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
        )
        .reset_index()
        .query("synthetic_model_type != 'None'")
    )

    combined_unnormalized["pre_post"] = (
        combined_unnormalized["preprocessing_strategy"]
        + " - "
        + combined_unnormalized["postprocessing_strategy"]
    )

    models: List[RegressionResultsWrapper] = []
    processing_estimates_dfs: List[pd.DataFrame] = []
    for task in combined_unnormalized["tasks"].unique():
        x_str = "AUROC" if task == "Classification" else "Spearman's Rank Correlation"
        tmp_df = combined_unnormalized.query("tasks == @task")

        (
            plot_density(tmp_df)
            + pn.labs(
                x=x_str,
                y="Density",
                color="Preprocessing - postprocessing",
                fill="Preprocessing - postprocessing",
                title=task,
            )
        ).save(
            plot_save_dir / f"density_{task}.png",
            height=7,
            width=10,
            dpi=400,
        )
        mdl = fit_lm(
            df=tmp_df,
            formula="mean ~ dataset_id * synthetic_model_type + preprocessing_strategy + postprocessing_strategy",
        )
        processing_estimates = get_processing_estimates(
            extract_estimate_and_p_value_from_lm(model=mdl)
        )
        processing_estimates["task"] = task
        processing_estimates_dfs.append(processing_estimates)
        models.append(mdl)

    processing_estimates_df = pd.concat(processing_estimates_dfs)
    processing_estimates_df["significant"] = (
        processing_estimates_df["p_value"] < 0.05
    ).astype(int)

    # table of variability by dataset
    variability_by_dataset = (
        aggregate_performance_across_tasks(
            classification_performance_df=performance_dfs.classification.query(
                "data_centric_method == @data_centric_method & classification_model_type == 'XGBClassifier' & metric == 'roc_auc'"
            ),
            model_selection_df=performance_dfs.model_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            feature_selection_df=performance_dfs.feature_selection.query(
                "data_centric_method == @data_centric_method"
            ),
            by_cols=[
                "dataset_id",
                "preprocessing_strategy",
                "postprocessing_strategy",
            ],
            normalize=False,
            normalize_index_col="",
        )
        .reset_index()
        .query("preprocessing_strategy != 'org_data'")
        .set_index(["dataset_id", "preprocessing_strategy", "postprocessing_strategy"])
    )
