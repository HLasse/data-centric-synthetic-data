from typing import List, Literal, Tuple

import openml
import pandas as pd


def get_task_X_y(task_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Get the X and y task from OpenML."""
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    return X, y


def get_openml_benchmark_suite_task_ids(suite_id: Literal[334, 337]) -> List[int]:
    """Get the same datasets as used in the tree-based vs NN paper.
     https://arxiv.org/pdf/2207.08815.pdf.
    SUITE_ID = 334 Classification on numerical and categorical features. (7 datasets)
    SUITE_ID = 337 Classification on numerical features (16 datasets)
    See https://github.com/LeoGrin/tabular-benchmark for more info.

    Their preprocessing:
    Not too small (>4 features, >3000 samples), not too easy, not high dimensional,
    real-world data, truncate to 10,000 samples, remove missing values, balanced and
    binarized classes (problem?), low cardinality categorical features (<20 items),
    and a few other criteria.
    """
    return openml.study.get_suite(suite_id).tasks  # type: ignore

