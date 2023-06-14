PREPROC_STRATEGY_MAPPING = {
    "org_data": "Real data",
    "baseline": "Baseline",
    "easy_hard": "Easy-hard",
    "easy_ambiguous_hard": "Easy-ambiguous-hard",
}

POSTPROC_STRATEGY_MAPPING = {
    "org_data": "Real data",
    "baseline": "Baseline",
    "easy_ambi": "No hard",
}

import pandas as pd

DATASET2ENUM_MAPPING = {
    361062: 1,
    361060: 2,
    361065: 3,
    361277: 4,
    361066: 5,
    361063: 6,
    361278: 7,
    361055: 8,
    361275: 9,
    361273: 10,
    361070: 11,
}

DATASET_ENUM2NAME_MAPPING = {
    1: "pol",
    2: "electricity",
    3: "magic telescope",
    4: "california housing",
    5: "bank marketing",
    6: "house 16H",
    7: "heloc",
    8: "credit",
    9: "default of credit card clients",
    10: "diabetes",
    11: "eye movements",
}


DATASET_MAPPING_DF = (
    pd.DataFrame(
        {
            361062: {
                "dataset_name": "pol",
                "dataset_id": 1,
                "n_features": 26,
                "n_samples": 10082,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361062",
            },
            361060: {
                "dataset_name": "electricity",
                "dataset_id": 2,
                "n_features": 7,
                "n_samples": 38474,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361060",
            },
            361065: {
                "dataset_name": "MagicTelescope",
                "dataset_id": 3,
                "n_features": 10,
                "n_samples": 13376,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361065",
            },
            361277: {
                "dataset_name": "california",
                "dataset_id": 4,
                "n_features": 8,
                "n_samples": 20634,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361277",
            },
            361066: {
                "dataset_name": "bank-marketing",
                "dataset_id": 5,
                "n_features": 7,
                "n_samples": 10578,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361066",
            },
            361063: {
                "dataset_name": "house_16H",
                "dataset_id": 6,
                "n_features": 16,
                "n_samples": 13488,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361063",
            },
            361278: {
                "dataset_name": "heloc",
                "dataset_id": 7,
                "n_features": 22,
                "n_samples": 10000,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361278",
            },
            361055: {
                "dataset_name": "credit",
                "dataset_id": 8,
                "n_features": 10,
                "n_samples": 16714,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361055",
            },
            361275: {
                "dataset_name": "default-of-credit-card-clients",
                "dataset_id": 9,
                "n_features": 20,
                "n_samples": 13272,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361275",
            },
            361273: {
                "dataset_name": "Diabetes130US",
                "dataset_id": 10,
                "n_features": 7,
                "n_samples": 71090,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361273",
            },
            361070: {
                "dataset_name": "eye_movements",
                "dataset_id": 11,
                "n_features": 20,
                "n_samples": 7608,
                "url": "https://openml.org/search?type=task&collections.id=337&sort=runs&id=361070",
            },
        }
    )
    .T.reset_index()
    .rename(columns={"index": "task_id"})
)

if __name__ == "__main__":
    df = DATASET_MAPPING_DF
    df = df.drop(columns=["dataset_id"])
    pd.set_option("display.max_colwidth", None)
    print(df.sort_values(by="task_id").to_latex(index=False))
