import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def show_result(clf, y_true, y_pred, figsize=(8, 6)):

    cm = confusion_matrix(y_true, y_pred, labels=y_pred)
    print(f"confusion matrix: \n {cm}")
    print(
        classification_report(y_true, y_pred, target_names=["Y", "N"], zero_division=0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=y_pred, ax=ax, colorbar=True
    )
    plt.title(f"{clf.__class__.__name__}")
    plt.show()
