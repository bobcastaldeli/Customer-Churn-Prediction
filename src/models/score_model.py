"""This module contains functions to score the model."""


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


def score_model(y_test, y_pred):
    """Score the model.

    params:
        y_test: test target
        y_pred: predicted target
    """
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, balanced_accuracy, precision, f1, roc_auc


def save_metrics(
    metrics_path, accuracy, balanced_accuracy, precision, f1, roc_auc
):
    """Save the metrics.

    params:
        metrics_path: path to the metrics output
        accuracy: accuracy score
        balanced_accuracy: balanced accuracy score
        precision: precision score
        f1: f1 score
        roc_auc: roc auc score
    """
    with open(metrics_path, "w") as file:
        file.write("Accuracy score: {:.2f}\n".format(accuracy))
        file.write(
            "Balanced accuracy score: {:.2f}\n".format(balanced_accuracy)
        )
        file.write("Precision score: {:.2f}\n".format(precision))
        file.write("F1 score: {:.2f}\n".format(f1))
        file.write("ROC AUC score: {:.2f}\n".format(roc_auc))
