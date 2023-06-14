from typing import Sequence

import numpy as np
import pandas as pd
import patchworklib as pw
import plotnine as pn
from plotnine.stats.stat_summary import mean_cl_boot


def remove_unused_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unused categories from a dataframe."""
    for col in df.select_dtypes("category"):
        df[col] = df[col].cat.remove_unused_categories()
    return df


def pointplot_with_error_bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    x_lab: str,
    y_lab: str,
) -> pn.ggplot:
    df = remove_unused_categories(df)
    return (
        pn.ggplot(
            df,
            pn.aes(x=x, y=y, color=color, fill=color),
        )
        + pn.geom_point(
            position=pn.position_dodge(width=0.5),
            stat="summary",
            fun_y=np.mean,
        )
        + pn.stat_summary(
            geom="errorbar",
            position=pn.position_dodge(width=0.5),
            fun_data="mean_cl_boot",
            # fun_ymin=lambda x: np.mean(x) - np.std(x),
            # fun_ymax=lambda x: np.mean(x) + np.std(x),
            width=0.3,
        )
        + pn.theme(
            legend_position="bottom",
            figure_size=(12, 8),
            legend_title=pn.element_blank(),
            legend_box_margin=50,
        )
        + pn.labs(x=x_lab, y=y_lab)
    )


def lineplot_with_error_bands(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    x_lab: str,
    y_lab: str,
) -> pn.ggplot:
    df = remove_unused_categories(df)
    return (
        pn.ggplot(
            df,
            pn.aes(x=x, y=y, color=color, fill=color, group=color),
        )
        + pn.geom_line(
            stat="summary",
            fun_y=np.mean,
        )
        + pn.stat_summary(
            geom="ribbon",
            fun_data="mean_cl_boot",
            alpha=0.2,
            color="none",
        )
        + pn.theme(
            legend_position="bottom",
            figure_size=(12, 8),
            legend_title=pn.element_blank(),
            legend_box_margin=50,
        )
        + pn.labs(x=x_lab, y=y_lab)
    )


# def lineplot_with_error_bands(
#     df: pd.DataFrame,
#     x: str,
#     y: str,
#     color: str,
#     x_lab: str,
#     y_lab: str,
# ) -> pn.ggplot:
#     ymin_ymax = df.groupby([x, color])["value"].apply(mean_cl_boot).reset_index()

#     return (
#         pn.ggplot(
#             df,
#         )
#         + pn.geom_ribbon(
#             data=ymin_ymax,
#             mapping=pn.aes(
#                 x=x,
#                 ymin="ymin",
#                 ymax="ymax",
#                 y="y",
#                 group=color,
#                 color=color,
#                 fill=color,
#             ),
#             alpha=0.2,
#         )
#         + pn.theme(
#             legend_position="bottom",
#             figure_size=(12, 8),
#             legend_title=pn.element_blank(),
#             legend_box_margin=50,
#         )
#         + pn.labs(x=x_lab, y=y_lab)
#     )


def facet_pointplot_with_error_bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    facet: str,
    x_lab: str,
    y_lab: str,
) -> pn.ggplot:
    df = remove_unused_categories(df)
    return (
        pn.ggplot(
            df,
            pn.aes(
                x=x,
                y=y,
                color=color,
                fill=color,
            ),
        )
        + pn.geom_point(
            position=pn.position_dodge(width=0.5),
            stat="summary",
            fun_y=np.mean,
        )
        + pn.stat_summary(
            geom="errorbar",
            position=pn.position_dodge(width=0.5),
            fun_data="mean_cl_boot",
            # fun_ymin=lambda x: np.mean(x) - np.std(x),
            # fun_ymax=lambda x: np.mean(x) + np.std(x),
            width=0.3,
        )
        + pn.facet_grid(facet)
        + pn.theme(
            legend_position="bottom",
            figure_size=(12, 8),
            legend_title=pn.element_blank(),
            legend_box_margin=50,
        )
        + pn.labs(x=x_lab, y=y_lab)
    )


def create_patchwork_grid(
    plots: Sequence[pn.ggplot],
    single_plot_dimensions: tuple[float, float],
    n_in_row: int,
    add_subpanels_letters: bool = True,
    fontsize="large",
) -> pw.Bricks:
    """Create a grid from a sequence of ggplot objects."""
    bricks = []

    for plot in plots:
        # Iterate here to catch errors while only a single plot is in scope
        # Makes debugging much easier
        bricks.append(pw.load_ggplot(plot, figsize=single_plot_dimensions))

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    rows = []
    current_row = []

    for i in range(len(bricks)):
        # Add the letter
        if add_subpanels_letters:
            bricks[i].set_index(alphabet[i].upper(), fontsize=fontsize)

        # Add it to the row
        current_row.append(bricks[i])

        row_is_full = i % n_in_row != 0 or n_in_row == 1
        all_bricks_used = i == len(bricks) - 1

        if row_is_full or all_bricks_used:
            # Rows should consist of two elements
            rows.append(pw.stack(current_row, operator="|"))
            current_row = []

    # Combine the rows
    patchwork = pw.stack(rows, operator="|")
    return patchwork
