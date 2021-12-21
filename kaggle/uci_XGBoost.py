import numpy as np
import pandas as pd
import xgboost as xgb
from rich import print
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import KMeansSMOTE
from utils import show_result


TRAIN_DIR = "uci_credit_card_default.csv"
TEST_DIR = "dataset/test.csv"
TEST_SIZE = 0.3
KMEANS_USE = False
TRAIN_TEST_SPLIT = False


def sta_encoder(x):
    if x == "duly":
        return 1
    elif x == "delay":
        return 0
    else:
        np.nan


def main():

    # load dataset
    train = pd.read_csv(TRAIN_DIR).drop(columns=["ID"])
    test = pd.read_csv(TEST_DIR).drop(columns=["ID"])

    # impute missing value

    # label encoder
    label_cols = ["SEX", "EDU", "MAR"]
    train[label_cols] = train[label_cols].apply(LabelEncoder().fit_transform)
    test[label_cols] = test[label_cols].apply(LabelEncoder().fit_transform)

    # label encoder for the target value
    y = train.PAY.values
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # label encoder for STA_1, STA_2 and STA_3
    for sta in ["STA_1", "STA_2", "STA_3"]:
        train[sta] = train[sta].apply(sta_encoder)
        test[sta] = test[sta].apply(sta_encoder)

    # assign data and label
    X, y = train.drop(columns="PAY").values, y
    sample_size, n_features = X.shape

    # K-means SMOTE overrsampling
    if KMEANS_USE:
        k = int(np.sqrt(sample_size / 2))
        sm = KMeansSMOTE(k_neighbors=k, sampling_strategy="auto")
        X, y = sm.fit_resample(X, y)

    # train test split

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        gamma=0.1,
        learning_rate=0.1,
        max_depth=200,
        reg_lambda=0.001,
        scale_pos_weight=1,
        subsample=0.9,
        colsample_bytree=0.5,
    )

    clf.fit(
        X,
        y,
        verbose=False,
        early_stopping_rounds=10,
        eval_metric="error",
        eval_set=[(X, y)],
    )

    # create submission
    y_pred = clf.predict(test.values)
    y_pred = ["Y" if y == 1 else "N" for y in y_pred]
    SUBMISSION = "dataset/sample_submission.csv"
    submission = pd.read_csv(SUBMISSION)
    submission["PAY"] = y_pred
    submission.to_csv(
        f"Classifier_{clf.__class__.__name__}, SMOTE_{KMEANS_USE}, Train_test_split_{TRAIN_TEST_SPLIT}.csv",
        index=False,
    )
    print(f"Submission Down!")


if __name__ == "__main__":
    main()
