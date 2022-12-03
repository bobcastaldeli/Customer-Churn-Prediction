# This module contains functions to build features for the model

import numpy as np
import pandas as pd


def casting_numerical(dataframe, numerical_feature):
    """Cast features to the correct type."""
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
    """Cast features to the correct type."""
    dataframe[categorical_feature] = dataframe[categorical_feature].astype(
        "object"
    )
    return dataframe


def change_no_service_to_no(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            dataframe[col] = dataframe[col].replace("No phone service", "No")
            dataframe[col] = dataframe[col].replace(
                "No internet service", "No"
            )
    return dataframe
