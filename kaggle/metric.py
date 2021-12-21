import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def show_result(clf, y_true, y_pred, label_encoder: LabelEncoder, figsize=(8, 6)):

    if not any(np.unique(y_pred) == -1):
        y_pred = label_encoder.inverse_transform(y_pred)
    y_true = label_encoder.inverse_transform(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
    print(f"confusion matrix: \n {cm}")
    print(
        classification_report(y_true, y_pred, target_names=["Y", "N"], zero_division=0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=label_encoder.classes_, ax=ax, colorbar=True
    )
    plt.title(f"{clf.__class__.__name__}")
    plt.show()
