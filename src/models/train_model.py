"""This script is used to train the model."""

import os
import sys
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from feature_engine.selection import DropFeatures
from feature_engine.creation import CombineWithReferenceFeature
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython train_model.py input-data-path ouput-model-path\n"
    )
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
train_input_path = os.path.join(input_path, "train.csv")
# model_output_path = os.path.join(output_path, "model.pkl")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    target = params["train_model"]["target"]


def train_model(train_path, model_path):
    """Train the model.

    params:
        train_path: path to the train data
        model_path: path to the model
    """
    logger.info("Loading data...")
    train = pd.read_csv(train_path)
    X_train = train.drop(target, axis=1)
    y_train = train[target]
    logger.info("Training model...")
    pipeline = pipeline = Pipeline(
        [
            ("drop_vars", DropFeatures(["customerID"])),
            (
                "tenure_combine",
                CombineWithReferenceFeature(
                    variables_to_combine=["MonthlyCharges", "TotalCharges"],
                    reference_variables=["tenure"],
                    operations=["div"],
                    new_variables_names=[
                        "tenureMonthlyRate",
                        "tenureTotalRate",
                    ],
                ),
            ),
            (
                "totalcharges_combine",
                CombineWithReferenceFeature(
                    variables_to_combine=["TotalCharges"],
                    reference_variables=["MonthlyCharges"],
                    operations=["div"],
                    new_variables_names=["RateCharge"],
                ),
            ),
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "num",
                            make_pipeline(
                                SimpleImputer(strategy="median"),
                            ),
                            make_column_selector(dtype_include=np.number),
                        ),
                        (
                            "cat",
                            make_pipeline(
                                SimpleImputer(strategy="most_frequent"),
                                TargetEncoder(),
                            ),
                            make_column_selector(dtype_include=["object"]),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=700,
                    learning_rate=0.0033570,
                    max_depth=5,
                    min_samples_split=6,
                    min_samples_leaf=21,
                    subsample=0.65,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    logger.info("Saving model...")
    # os.makedirs(sys.argv[2], exist_ok=True)
    with open(model_path, "wb") as fd:
        pickle.dump(pipeline, fd)


if __name__ == "__main__":
    train_model(train_input_path, output_path)
