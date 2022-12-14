"""The split_data function takes the path to the raw data, the test size, and
the random state as arguments.

It then loads the data, splits it into train and test sets, and saves
the results to the data folder.
"""

import os
import sys
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


input_path, output_path = sys.argv[1], sys.argv[2]


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    test_size = params["split_data"]["test_size"]
    stratify = params["split_data"]["stratify"]
    random_state = params["split_data"]["random_state"]


def split_data(input_path, output_path):
    """Split data into train and test sets.

    params:
        data_path: path to the raw data
        test_size: size of the test set
        stratify: column to stratify the split
        random_state: random state seed
    """
    logger.info("Loading data...")
    df = pd.read_csv(os.path.join(input_path, "telco-customer-churn.zip"))
    logger.info("Splitting data...")
    train, test = train_test_split(
        df,
        test_size,
        df[stratify],
        random_state,
    )
    logger.info("Saving data...")
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)


if __name__ == "__main__":
    split_data(input_path, output_path)
