"""This script is used to predict the output of the model."""


import os
import sys
import logging
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    plot_precision_recall_curve,
    plot_roc_curve,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 4 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython evaluate_model.py input_data-path train-model-path"
        " predictions-output-path\n"
    )
    sys.exit(1)


input_path, model_path, prediction_path = sys.argv[1], sys.argv[2], sys.argv[3]
test_input_path = os.path.join(input_path, "test.csv")
model_input_path = os.path.join(model_path, "model.pkl")
metrics_path = os.path.join(prediction_path, "metrics.txt")
predict_path = os.path.join(prediction_path, "predictions.csv")
plot_path = os.path.join(prediction_path, "metrics_reports.png")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    target = params["evaluate_model"]["target"]


def predict_model(test_path, model, pred_path):
    """Predict the output of the model.

    params:
        test_path: path to the test data
        model: path to the model
        pred_path: path to the predictions output
    """
    logger.info("Loading data...")
    test = pd.read_csv(test_path)
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    logger.info("Loading model...")
    with open(model, "rb") as file:
        model = pickle.load(file)
    logger.info("Predicting output...")
    y_pred = model.predict(X_test)
    os.makedirs(sys.argv[3], exist_ok=True)
    logger.info("Computing metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    with open(metrics_path, "w") as file:
        file.write('Accuracy score: {}\n'.format(accuracy))
        file.write('Balanced Accuracy score: {}\n'.format(balanced_accuracy))
        file.write('Precision score: {}\n'.format(precision))
        file.write('F1 score: {}\n'.format(f1))
        file.write('ROC AUC score: {}\n'.format(roc_auc))
    logger.info("Generating plots...")
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    plot_roc_curve(model, X_test, y_test, ax=ax[0])
    plot_precision_recall_curve(model, X_test, y_test, ax=ax[1])
    for i, plot in enumerate([plot_path]):
        fig.savefig(plot)
    logger.info("Saving output...")
    output = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    output.to_csv(pred_path, index=False)


if __name__ == "__main__":
    predict_model(test_input_path, model_input_path, predict_path)
