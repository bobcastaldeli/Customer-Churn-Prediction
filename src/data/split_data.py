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


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython split_data.py data-dir-path ouput-dir-path\n")
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
input_path = os.path.join(input_path, "telco-customer-churn.zip")
train_output_path = os.path.join(output_path, "train.csv")
test_output_path = os.path.join(output_path, "test.csv")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    test_size = params["split_data"]["test_size"]
    stratify = params["split_data"]["stratify"]
    random_state = params["split_data"]["random_state"]


def split_data(input_path, train_path, test_path):
    """Split data into train and test sets.

    params:
        input_path: path to the raw data
        train_path: path to the train data
        test_path: path to the test data
    """
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    logger.info("Splitting data...")
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify],
        random_state=random_state,
    )
    logger.info("Saving data...")
    os.makedirs(sys.argv[2], exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


if __name__ == "__main__":
    split_data(input_path, train_output_path, test_output_path)
