# This module contains functions to build features for the model

import logging
import argparse
import pandas as pd
from src.features.casting_vars import casting_numerical, casting_categorical


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def build_features(train_path, test_path):
    """Build features from raw data."""
    logger.info("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info("Building features...")
    train, test = map(
        lambda dataframe: casting_numerical(
            dataframe, "TotalCharges", [train, test]
        )
    )
    train, test = map(
        lambda dataframe: casting_categorical(
            dataframe, "SeniorCitizen", [train, test]
        )
    )
    train, test = map(
        lambda dataframe: dataframe.dropna(),
        [train, test],
    )
    logger.info("Saving data...")
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/raw/train.csv")
    parser.add_argument("--test_path", default="data/raw/test.csv")
    args = parser.parse_args()

    build_features(args.train_path, args.test_path)
