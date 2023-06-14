"""Script to evaluate the experiment for figure1. This is done by plotting the proportion of
easy examples against the inverse kl divergence and classification performance."""


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotnine as pn

from application.constants import DATA_DIR, RESULTS_DIR
from application.figure1.run_figure1_exp import POCExperiment
from data_centric_synth.evaluation.plotting import create_patchwork_grid
from data_centric_synth.serialization.serialization import load_from_pickle


def load_poc_experiments(experiment_dir: Path) -> List[POCExperiment]:
    """Load the POC experiments from the experiment directory."""
    experiments: List[POCExperiment] = []
    for file in experiment_dir.iterdir():
        if file.name.endswith(".pkl"):
            experiments.extend(load_from_pickle(file))

    return experiments


def poc_experiments_to_dataframe(experiments: List[POCExperiment]) -> pd.DataFrame:
    """Convert the POC experiments to a dataframe with a column for
    data_centric_method, synthetic_model, dataset, and number of
    easy, ambiguous and hard examples."""
    exp_dict = [
        {
            "data_centric_method": exp.data_centric_method,
            "synthetic_model": exp.synthetic_model,
            "dataset": exp.dataset,
            "n_easy": len(exp.indices.easy),
            "n_ambiguous": len(exp.indices.ambiguous)
            if exp.indices.ambiguous is not None
            else 0,
            "n_hard": len(exp.indices.hard),
            "inv_kl_divergence": exp.inv_kl_divergence,
            "auc": exp.auc,
        }
        for exp in experiments
    ]
    df = pd.DataFrame.from_dict(exp_dict)
    # add proportions
    df["Easy"] = df["n_easy"] / (df["n_easy"] + df["n_ambiguous"] + df["n_hard"])
    df["Ambiguous"] = df["n_ambiguous"] / (
        df["n_easy"] + df["n_ambiguous"] + df["n_hard"]
    )
    df["Hard"] = df["n_hard"] / (df["n_easy"] + df["n_ambiguous"] + df["n_hard"])
    return df


if __name__ == "__main__":
    for dataset in ["covid", "adult"]:
        # experiments = load_poc_experiments(Path(RESULTS_DIR) / "poc" / dataset)
        experiments = load_poc_experiments(Path(DATA_DIR) / "figure1" / dataset)

        df = poc_experiments_to_dataframe(experiments)
        plot_save_dir = RESULTS_DIR / "figure1" / dataset / "plots"
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        # make data segment categorical

        # df = df.query("data_centric_method == 'cleanlab'")
        df["is_real"] = np.where(df["synthetic_model"] == "Real data", True, False)
        df["synthetic_model"] = df["synthetic_model"].replace(
            {
                "bayesian_network": "bn",
            }
        )

        # convert to long format to have one column for the proportion and one for
        # the data segment
        df_long = pd.melt(
            df,
            id_vars=["synthetic_model", "dataset", "data_centric_method"],
            value_vars=["Easy", "Ambiguous", "Hard"],
            var_name="data_segment",
            value_name="proportion",
        )
        df_long["data_segment"] = pd.Categorical(
            df_long["data_segment"],
            categories=["Easy", "Ambiguous", "Hard"],
        )

        # plot the proportion of easy, ambiguous and hard examples in the synthetic
        # data against the proportion of easy, ambiguous and hard examples in the
        # original data as a stacked bar plot
        (
            pn.ggplot(
                df_long,
                pn.aes(
                    x="synthetic_model",
                    y="proportion",
                    fill="data_segment",
                    color="data_segment",
                ),
            )
            + pn.geom_bar(stat="identity", position="stack")
            + pn.facet_grid("data_centric_method ~ .")
            + pn.theme(
                legend_position="bottom",
                figure_size=(12, 8),
                legend_title=pn.element_blank(),
                # legend_box_margin=80,
                legend_box_spacing=0.5,
            )
            + pn.labs(x="Synthetic Model", y="Proportion")
        )

        # plot proportion hard as line plot
        (
            pn.ggplot(
                df,
                pn.aes(
                    x="synthetic_model",
                    y="Hard",
                    group="dataset",
                ),
            )
            + pn.geom_line()
            + pn.geom_point()
            + pn.facet_grid("data_centric_method ~ .", scales="free_y")
            + pn.theme(
                figure_size=(8, 6),
                # rotate x-axis labels
                axis_text_x=pn.element_text(angle=45, hjust=1),
            )
            + pn.labs(x="Synthetic Model", y="Proportion Hard")
        )

        # plot proportion hard as bar chart
        (
            pn.ggplot(
                df,
                pn.aes(
                    x="synthetic_model",
                    y="Hard",
                    fill="is_real",
                ),
            )
            + pn.geom_bar(stat="identity")
            + pn.facet_grid("data_centric_method ~ .", scales="free_y")
            # color red and black
            + pn.scale_fill_manual(values=["#808080", "#cf1920"])
            + pn.theme(
                figure_size=(8, 6),
                # rotate x-axis labels
                axis_text_x=pn.element_text(angle=45, hjust=1),
                legend_position="none",
            )
            + pn.labs(x="Synthetic Model", y="Proportion Hard")
        ).save(plot_save_dir / "facet_proportion_hard.png", dpi=300)

        # plot proportion hard as bar chart only for cleanlab
        prop_hard_cleanlab = (
            pn.ggplot(
                df[df["data_centric_method"] == "cleanlab"],
                pn.aes(
                    x="synthetic_model",
                    y="Easy",
                    fill="is_real",
                    color="is_real",
                ),
            )
            + pn.geom_point(size=10)
            # color red and black
            + pn.scale_fill_manual(values=["#808080", "#cf1920"])
            + pn.scale_color_manual(values=["#808080", "#cf1920"])
            # + pn.ylim(0, 0.4)
            + pn.theme_minimal()
            + pn.theme(
                # figure_size=(8, 6),
                # rotate x-axis labels
                axis_text_x=pn.element_text(angle=45, hjust=1),
                legend_position="none",
                text=pn.element_text(size=22),
                panel_grid_major=pn.element_line(size=2),
            )
            + pn.labs(x="Synthetic Model", y="Proportion Easy")
        )
        # prop_hard_cleanlab.save(plot_save_dir / "proportion_hard_cleanlab.png", dpi=300)

        # plot inv_kl_divergence as bar chart only for cleanlab
        inv_kl_divergence_cleanlab = (
            pn.ggplot(
                df[df["data_centric_method"] == "cleanlab"],
                pn.aes(
                    x="synthetic_model",
                    y="inv_kl_divergence",
                    fill="is_real",
                    color="is_real",
                ),
            )
            + pn.geom_point(size=10)
            # color red and black
            + pn.scale_fill_manual(values=["#808080", "#cf1920"])
            + pn.scale_color_manual(values=["#808080", "#cf1920"])
            + pn.ylim(0.8, 1)
            + pn.theme_minimal()
            + pn.theme(
                # figure_size=(8, 6),
                # rotate x-axis labels
                axis_text_x=pn.element_text(angle=45, hjust=1),
                legend_position="none",
                text=pn.element_text(size=22),
                panel_grid_major=pn.element_line(size=2),
            )
            + pn.labs(x="Synthetic Model", y="Inv. KL Divergence")
        )
        # ploto auc as point plot only for cleanlab
        auc_cleanlab = (
            pn.ggplot(
                df[df["data_centric_method"] == "cleanlab"],
                pn.aes(
                    x="synthetic_model",
                    y="auc",
                    fill="is_real",
                    color="is_real",
                ),
            )
            + pn.geom_point(size=10)
            # color red and black
            + pn.scale_fill_manual(values=["#808080", "#cf1920"])
            + pn.scale_color_manual(values=["#808080", "#cf1920"])
            # + pn.ylim(0.8, 1)
            + pn.theme_minimal()
            + pn.theme(
                # figure_size=(8, 6),
                # rotate x-axis labels
                axis_text_x=pn.element_text(angle=45, hjust=1),
                legend_position="none",
                text=pn.element_text(size=22),
                panel_grid_major=pn.element_line(size=2),
            )
            + pn.labs(x="Synthetic Model", y="AUC")
        )

        combined = create_patchwork_grid(
            [inv_kl_divergence_cleanlab, auc_cleanlab, prop_hard_cleanlab],
            single_plot_dimensions=(5, 3),
            n_in_row=3,
            fontsize="xx-large",
        )

        combined.savefig(
            plot_save_dir / "proportion_hard_inv_kl_divergence_cleanlab.png", dpi=300
        )

        # plot proportions as line plot facet by data_segment
        (
            pn.ggplot(
                df_long,
                pn.aes(
                    x="synthetic_model",
                    y="proportion",
                    group="dataset",
                    color="data_segment",
                ),
            )
            + pn.geom_point()
            + pn.geom_line()
            + pn.facet_grid("data_segment~data_centric_method", scales="free_y")
            + pn.theme(
                figure_size=(12, 8),
                axis_text_x=pn.element_text(angle=20, hjust=1),
                legend_position="bottom",
                legend_title=pn.element_blank(),
                legend_box_spacing=0.8,
            )
            + pn.labs(x="Synthetic Model", y="Proportion")
        ).save(plot_save_dir / "poc_proportions.png", dpi=300)
