"""This script is used to predict the output of the model."""


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
        "\tpython predict_model.py input-data-path ouput-model-path\n"
    )
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
train_input_path = os.path.join(input_path, "train.csv")
# model_output_path = os.path.join(output_path, "model.pkl")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    target = params["evaluate_model"]["target"]


def predict_model(test_path, model_path, output_path):
    """Predict the output of the model.

    params:
        test_path: path to the test data
        model_path: path to the model
        output_path: path to the output
    """
    logger.info("Loading data...")
    test = pd.read_csv(test_path)
    X_test = test.drop("Churn", axis=1)
    y_test = test["Churn"]
    logger.info("Loading model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Predicting output...")
    y_pred = model.predict(X_test)
    logger.info("Saving output...")
    output = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, help="path to the test data")
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument(
        "--output_path", type=str, help="path to the output of the model"
    )
    args = parser.parse_args()
    predict_model(args.test_path, args.model_path, args.output_path)
