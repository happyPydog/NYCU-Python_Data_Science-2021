import sys
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import KMeansSMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from metric import show_result


TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"
UCI_DIR = "uci_credit_card_default.csv"


def main():

    df_uci = pd.read_csv(UCI_DIR).drop(columns="ID")
    df_test = pd.read_csv(TEST_DIR).drop(columns="ID")

    # # label encoder
    # train_label = df_uci["PAY"].values
    # test_label = df_uci["PAY"].values
    # le = LabelEncoder()
    # le.fit(label)
    # label = le.transform(label)

    # label encoder
    label_col = ["STA_1", "STA_2", "STA_3"]
    df[label_col] = df[label_col].apply(LabelEncoder().fit_transform)

    # label encoder for the target values
    le = LabelEncoder()
    le.fit(df["PAY"].values)
    df["PAY"] = le.transform(df["PAY"].values)

    # train test split
    X, y = df.drop(columns="PAY").values, df["PAY"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    sample_size, _ = X_train.shape
    print(
        f"{X_train.shape = }, {y_train.shape = }, {X_test.shape = }, {y_test.shape = }"
    )

    # min-max scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # k-means SMOTE
    k = int(np.sqrt(sample_size / 2))
    sm = KMeansSMOTE(k_neighbors=k)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train).items()))

    # model
    clf = MLPClassifier(
        hidden_layer_sizes=55,
        activation="relu",
        solver="sgd",
        alpha=0.0001,
        batch_size=1000,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Training Score: {clf.score(X_train, y_train):.2f}")
    print(f"Testing Score: {clf.score(X_test, y_test):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
    show_result(clf, y_test, y_pred, le)


if __name__ == "__main__":
    main()
