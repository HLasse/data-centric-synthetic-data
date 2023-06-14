import numpy as np
import openml
import pandas as pd


def load_adult_dataset() -> pd.DataFrame:
    """
    Loads and preprocesses the adult dataset.
    Removes all the rows with missing values and maps the categorical variables
    to numerical values.

    Returns:
        A dataframe with the processed adult dataset.
    """

    def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values
        Args:
          df: The dataframe to be processed
        Returns:
          a dataframe after the mapping
        """

        salary_map = {"<=50K": 1, ">50K": 0}
        df["salary"] = df["class"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).astype(int)

        df["native-country"] = df["native-country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        for col in ["native-country", "workclass", "occupation"]:
            df[col] = df[col].astype(str)

        df = df.dropna(how="any")

        df.loc[
            df["native-country"] != "United-States",
            "native-country",
        ] = "Non-US"

        df.loc[
            df["native-country"] == "United-States",
            "native-country",
        ] = "US"

        df["native-country"] = (
            df["native-country"].map({"US": 1, "Non-US": 0}).astype(int)
        )

        df["marital-status"] = df["marital-status"].replace(
            [
                "Divorced",
                "Married-spouse-absent",
                "Never-married",
                "Separated",
                "Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            ["Married-AF-spouse", "Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            "Unmarried": 0,
            "Wife": 1,
            "Husband": 2,
            "Not-in-family": 3,
            "Own-child": 4,
            "Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            "White": 0,
            "Amer-Indian-Eskimo": 1,
            "Asian-Pac-Islander": 2,
            "Black": 3,
            "Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if (
                x["workclass"] == "Federal-gov"
                or x["workclass"] == "Local-gov"
                or x["workclass"] == "State-gov"
            ):
                return "govt"
            elif x["workclass"] == "Private":
                return "private"
            elif (
                x["workclass"] == "Self-emp-inc" or x["workclass"] == "Self-emp-not-inc"
            ):
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df = df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
                "class",
            ],
            axis=1,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    # dataset id 1590 is the adult dataset (https://www.openml.org/search?type=data&sort=runs&id=1590&status=active)
    df = openml.datasets.get_dataset(1590).get_data(dataset_format="dataframe")[0]

    df = process_dataset(df)  # type: ignore

    return df
