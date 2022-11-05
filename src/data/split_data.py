"""The split_data function takes the path to the raw data, the test size, and
the random state as arguments.

It then loads the data, splits it into train and test sets, and saves
the results to the data folder.
"""

import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def split_data(data_path, test_size, random_state):
    """Split data into train and test sets."""
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    logger.info("Splitting data...")
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    logger.info("Saving data...")
    train.to_csv("data/raw/train.csv", index=False)
    test.to_csv("data/raw/test.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/raw/telco-customer-churn.zip"
    )
    parser.add_argument("--test_size", default=0.2)
    parser.add_argument("--random_state", default=42)
    args = parser.parse_args()

    split_data(args.data_path, args.test_size, args.random_state)
