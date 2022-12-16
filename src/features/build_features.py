"""This module contains functions to build features for the model."""

import os
import sys
import logging
import yaml
import pandas as pd
from casting_vars import (
    map_target,
    casting_numerical,
    casting_categorical,
    change_no_service_to_no,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython build_features.py input-data-path ouput-data-path\n"
    )
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
train_input_path = os.path.join(input_path, "train.csv")
test_input_path = os.path.join(input_path, "test.csv")
train_output_path = os.path.join(output_path, "train.csv")
test_output_path = os.path.join(output_path, "test.csv")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    target = params["build_features"]["target"]
    categorical_feature = params["build_features"]["cat_feature"]
    numerical_feature = params["build_features"]["num_feature"]


def build_features(input_train, input_test, output_train, output_test):
    """Build features from raw data.

    params:
        input_train: path to the raw train data
        input_test: path to the raw test data
        output_train: path to the processed train data
        output_test: path to the processed test data
    """
    logger.info("Loading data...")
    train = pd.read_csv(input_train)
    test = pd.read_csv(input_test)
    logger.info("Building features...")
    # apply map_target function to target in train and test at the same time
    train, test = map(
        lambda dataframe: map_target(dataframe, target),
        [train, test],
    )

    # apply casting_numerical function to numerical_feature in train and test at the same time
    train, test = map(
        lambda dataframe: casting_numerical(dataframe, numerical_feature),
        [train, test],
    )

    # apply casting_categorical function to categorical_feature in train and test at the same time
    train, test = map(
        lambda dataframe: casting_categorical(dataframe, categorical_feature),
        [train, test],
    )

    train, test = map(
        lambda dataframe: dataframe.dropna(),
        [train, test],
    )
    train, test = map(
        lambda dataframe: change_no_service_to_no(dataframe),
        [train, test],
    )
    logger.info("Saving data...")
    os.makedirs(sys.argv[2], exist_ok=True)
    train.to_csv(output_train, index=False)
    test.to_csv(output_test, index=False)


if __name__ == "__main__":
    build_features(
        train_input_path, test_input_path, train_output_path, test_output_path
    )
