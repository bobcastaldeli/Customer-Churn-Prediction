"""This module contains functions to plot the metrics of model evaluation."""


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)


def plot_confusion_matrix(model, y_test, y_pred, output_path):
    """Plot the confusion matrix.

    params:
        model: model
        y_test: test target
        y_pred: predicted target
        output_path: path to the confusion matrix output
    """
    cm = ConfusionMatrixDisplay.from_estimator(model, y_test, y_pred)
    cm.figure_.savefig(output_path)


def plot_precision_recall_curve(model, y_test, y_pred, output_path):
    """Plot the precision recall curve.

    params:
        model: model
        y_test: test target
        y_pred: predicted target
        output_path: path to the precision recall curve output
    """
    pr = PrecisionRecallDisplay.from_estimator(model, y_test, y_pred)
    pr.figure_.savefig(output_path)


def plot_roc_auc(model, y_test, y_pred, output_path):
    """Plot the roc auc curve.

    params:
        model: model
        y_test: test target
        y_pred: predicted target
        output_path: path to the roc auc curve output
    """
    roc = RocCurveDisplay.from_estimator(model, y_test, y_pred)
    roc.figure_.savefig(output_path)
