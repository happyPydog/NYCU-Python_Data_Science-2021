import sys
from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb
from rich import print
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import KMeansSMOTE
from sklearn.metrics import roc_auc_score
from metric import show_result

import torch
import torch.nn as nn


TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"
RANDOM_SEED = 5566
TEST_SIZE = 0.3
BATCH_SIZE = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
EPOCHS = 1000


class LinerNet(nn.Module):
    """Linear neural network."""

    def __init__(self, in_features: int):
        super(LinerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 66),
            nn.ReLU(),
            nn.Linear(66, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


def main():

    # load data
    train = pd.read_csv(TRAIN_DIR).drop(columns=["ID", "STA_1", "STA_2", "STA_3"])
    test = pd.read_csv(TEST_DIR).drop(columns=["ID", "STA_1", "STA_2", "STA_3"])

    # label encoder target value
    le = LabelEncoder()
    le.fit(train["PAY"].values)
    train["PAY"] = le.transform(train["PAY"].values)

    # Assign data and target value
    X, y = train.drop(columns="PAY"), train["PAY"].values

    # impute missing value
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_most_frequent.fit(X.values)
    train = pd.DataFrame(imp_most_frequent.transform(X.values), columns=X.columns)
    test = pd.DataFrame(imp_most_frequent.transform(test.values), columns=test.columns)

    # check column type
    col_numerical = {
        "AGE": int,
        "CRE": int,
        "BILL_1": int,
        "BILL_2": int,
        "BILL_3": int,
        "AMT_1": int,
        "AMT_2": int,
        "AMT_3": int,
    }
    train = train.astype(col_numerical)
    test = test.astype(col_numerical)

    # onehot-encoder
    df_dummy = pd.get_dummies(train)
    train = pd.DataFrame(df_dummy)
    df_dummy = pd.get_dummies(test)
    test = pd.DataFrame(df_dummy)

    # pd.DataFrame to array
    X = train.values
    sample_size, n_features = X.shape

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Original: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}")

    # Min-Max nomralize
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # model
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=10,
        use_label_encoder=False,
        learning_rate=0.2,
        max_depth=200,
    )

    clf.fit(
        X_train,
        y_train,
        verbose=False,
        eval_metric="logloss",
        eval_set=[(X_val, y_val)],
    )

    y_pred = clf.predict(X_val)
    print(f"Training Score: {clf.score(X_train, y_train):.2f}")
    print(f"Testing Score: {clf.score(X_val, y_val):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_pred):.2f}")
    show_result(clf, y_val, y_pred, le)


if __name__ == "__main__":
    main()
