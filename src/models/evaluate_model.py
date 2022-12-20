"""This script is used to predict the output of the model."""


import os
import sys
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from feature_engine.selection import DropFeatures
from feature_engine.creation import CombineWithReferenceFeature
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    plot_precision_recall_curve,
    plot_roc_curve,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython evaluate_model.py input-data-path train-model-path"
        " ouput-data-path output-report-path\n"
    )
    sys.exit(1)


input_path, model_path, prediction_path, report_path = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
)
test_input_path = os.path.join(input_path, "test.csv")
# model = os.path.join(model_path, "model.pkl")
predict_path = os.path.join(prediction_path, "predict.csv")
# classf_report_path = os.path.join(report_path, "classification_report.png")
# rocauc_path = os.path.join(report_path, "roc_auc_curve.png")
# pr_path = os.path.join(report_path, "precision_recall_curve.png")
# model_output_path = os.path.join(output_path, "model.pkl")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    target = params["evaluate_model"]["target"]


def predict_model(test_path, model, pred_path, reports_path):
    """Predict the output of the model.

    params:
        test_path: path to the test data
        model: path to the model
        pred_path: path to the output
        reports_path: path to the reports
    """
    logger.info("Loading data...")
    test = pd.read_csv(test_path)
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    logger.info("Loading model...")
    with open(model, "rb") as f:
        model = pickle.load(f)
    logger.info("Predicting output...")
    y_pred = model.predict(X_test)
    # save predictions
    logger.info("Saving output...")
    os.makedirs(sys.argv[2], exist_ok=True)
    output = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    output.to_csv(pred_path, index=False)
    # print classification report
    print(classification_report(y_test, y_pred))
    plt.savefig(os.path.join(reports_path, "classification_report.png"))
    # plot roc auc curve
    plot_roc_curve(model, X_test, y_test)
    plt.savefig(os.path.join(reports_path, "roc_auc_curve.png"))
    # plot precision recall curve
    plot_precision_recall_curve(model, X_test, y_test)
    plt.savefig(os.path.join(reports_path, "precision_recall_curve.png"))


if __name__ == "__main__":
    predict_model(test_input_path, model_path, predict_path, report_path)
