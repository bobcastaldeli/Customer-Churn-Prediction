"""This module contains functions to build features for the model."""

import numpy as np
import pandas as pd


def map_target(dataframe, target):
    """Map target to 0 and 1.

    params:
        dataframe: dataframe to map
        target: target to map
    """
    dataframe[target] = dataframe[target].map({"No": 0, "Yes": 1})
    return dataframe


def casting_numerical(dataframe, numerical_feature):
    """Cast features to the correct type.

    params:
        dataframe: dataframe to cast
        numerical_feature: numerical feature to cast
    """
    dataframe[numerical_feature] = dataframe[numerical_feature].apply(
        lambda dataframe: str(dataframe).replace(",", "."),
    )
    dataframe[numerical_feature] = pd.to_numeric(
        dataframe[numerical_feature], errors="coerce"
    )
    dataframe[numerical_feature] = dataframe[numerical_feature].astype(
        "float64"
    )
    dataframe[numerical_feature] = dataframe[numerical_feature].replace(
        "", np.nan
    )
    return dataframe


def casting_categorical(dataframe, categorical_feature):
    """Cast features to the correct type.

    params:
        dataframe: dataframe to cast
        categorical_feature: categorical feature to cast
    """
    dataframe[categorical_feature] = dataframe[categorical_feature].astype(
        "object"
    )
    return dataframe


def change_no_service_to_no(dataframe):
    """Change "No phone service" and "No internet service" to "No".

    params:
        dataframe: dataframe to change
    """
    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            dataframe[col] = dataframe[col].replace("No phone service", "No")
            dataframe[col] = dataframe[col].replace(
                "No internet service", "No"
            )
    return dataframe
