"""Help function interface."""
import time
from typing import Callable, Any
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

COLORS_LIST = ["tab:blue", "tab:orange"]
TARTGET_NAME = ["Y", "N"]
COLOR_TICKS = np.arange(-1, 2)


def show_result(clf, y_true, y_pred, le, save=False, figsize=(8, 6)):
    """Show result of Confusion Matrix."""

    y_true = le.inverse_transform(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
    print(f"confusion matrix: \n {cm}")
    print(
        classification_report(
            y_true, y_pred, target_names=TARTGET_NAME, zero_division=0
        )
    )

    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=le.classes_, ax=ax, colorbar=True
    )
    plt.title(f"{clf.__class__.__name__}")
    plt.show()
    if save:
        plt.savefig(f"./figure/{clf.__class__.__name__}")


def plot_DicisionRegions(clf, X, y, figsize=(8, 6), num=100, extend=1):

    plt.figure(figsize=figsize)
    x1 = np.linspace(np.min(X[:, 0]) - extend, np.max(X[:, 0]) + extend, num=num)
    x2 = np.linspace(np.min(X[:, 1]) - extend, np.max(X[:, 1]) + extend, num=num)
    x1, x2 = np.meshgrid(x1, x2)
    xy = np.c_[x1.flatten(), x2.flatten()]
    z = clf.predict(xy).reshape(x1.shape)

    # plot contourf
    plt.contourf(
        x1, x2, z, levels=np.arange(-1, 2), colors=["tab:blue", "tab:orange"], alpha=0.5
    )

    for color, label, target_name in zip(
        ["tab:blue", "tab:orange"], [0, 1], ["Y", "N"]
    ):
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            color=color,
            alpha=0.8,
            lw=2,
            label=target_name,
        )
    plt.legend(loc="best")
    plt.title(f"{clf.__class__.__name__}")
    plt.show()


def impute_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    """Random Sample imputation."""

    categorical_col = [col for col, df_type in zip(df, df.dtypes) if df_type == object]

    for col in categorical_col:

        # null sample size in the column
        na_in_col = df[col].isnull()

        # NAN index
        na_idx = np.where(na_in_col)[0]

        # sample and replace NAN
        random_sample = df[col].dropna().sample(len(na_idx))
        random_sample.index = df[na_in_col].index
        df.loc[na_in_col, col] = random_sample

    return df


class RunTime:
    """Run time decorator."""

    def __call__(self, func) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} Run times: {end - start}/s")

        return wrapper
