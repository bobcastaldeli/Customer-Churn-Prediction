"""This script is used to train the model."""


import logging
import argparse
import pickle
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


def train_model(train_path, model_path):
    """Train the model.

    params:
        train_path: path to the train data
        model_path: path to the model
    """
    logger.info("Loading data...")
    train = pd.read_csv(train_path)
    X_train = train.drop("Churn", axis=1)
    y_train = train["Churn"]
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
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/raw/train.csv")
    parser.add_argument("--model_path", default="models/model.pkl")
    args = parser.parse_args()

    train_model(args.train_path, args.model_path)
