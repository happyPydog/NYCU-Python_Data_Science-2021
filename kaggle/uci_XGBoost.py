import numpy as np
import pandas as pd
import xgboost as xgb
from rich import print
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import KMeansSMOTE


TRAIN_DIR = "uci_credit_card_default.csv"
TEST_DIR = "dataset/test.csv"
TEST_SIZE = 0.3
TRAIN_TEST_SPLIT = False
KMEANS_USE = False


CLF = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    scale_pos_weight=3.34,
    learning_rate=0.01,
    max_depth=2000,
    reg_lambda=0.001,
)


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

    # impute missing values for testing data
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_most_frequent.fit(test.values)
    test = pd.DataFrame(imp_most_frequent.transform(test.values), columns=test.columns)

    # label encoder
    label_cols = ["SEX", "EDU", "MAR", "STA_1", "STA_2", "STA_3"]
    train[label_cols] = train[label_cols].apply(LabelEncoder().fit_transform)
    test[label_cols] = test[label_cols].apply(LabelEncoder().fit_transform)

    # label encoder for the target value
    y = train.PAY.values
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # assign data and label
    X, y = train.drop(columns="PAY").values, y
    X_test = test.values
    sample_size, _ = X.shape

    # train test split
    if TRAIN_TEST_SPLIT:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=True, stratify=y
        )
        print(f"{X_train.shape= }, {y_train.shape= }, {X_val.shape= }, {y_val.shape= }")

        # K-means SMOTE overrsampling
        if KMEANS_USE:
            k = int(np.sqrt(sample_size / 2))
            sm = KMeansSMOTE(
                k_neighbors=k, cluster_balance_threshold=0.2, sampling_strategy="auto"
            )
            X_train, y_train = sm.fit_resample(X_train, y_train)

        CLF.fit(
            X_train,
            y_train,
            verbose=False,
            early_stopping_rounds=10,
            eval_metric="error",
            eval_set=[(X_val, y_val)],
        )
        y_pred = CLF.predict(X_val)
        # print metric
        print(f"Training Score {CLF.score(X_train, y_train):.2f}")
        print(f"Testing Score {CLF.score(X_val, y_val):.2f}")
        print(f"ROC AUC  : {roc_auc_score(y_val, y_pred)}")

        # create submission
        y_pred = CLF.predict(X_test)
        y_pred = ["Y" if y == 1 else "N" for y in y_pred]
        SUBMISSION = "dataset/sample_submission.csv"
        submission = pd.read_csv(SUBMISSION)
        submission["PAY"] = y_pred
        submission.to_csv(
            f"Classifier_{CLF.__class__.__name__}, SMOTE_{KMEANS_USE}, Train_test_split_{TRAIN_TEST_SPLIT}.csv",
            index=False,
        )
        print(f"Submission Down!")
        print(f"File name: {CLF.__class__.__name__} {KMEANS_USE} {TRAIN_TEST_SPLIT}")

    else:
        # K-means SMOTE overrsampling
        if KMEANS_USE:
            k = int(np.sqrt(sample_size / 2))
            sm = KMeansSMOTE(k_neighbors=k, sampling_strategy="auto")
            X, y = sm.fit_resample(X, y)

        CLF.fit(
            X,
            y,
            verbose=False,
            early_stopping_rounds=10,
            eval_metric="error",
            eval_set=[(X, y)],
        )

        # create submission
        y_pred = CLF.predict(test.values)
        y_pred = ["Y" if y == 1 else "N" for y in y_pred]
        SUBMISSION = "dataset/sample_submission.csv"
        submission = pd.read_csv(SUBMISSION)
        submission["PAY"] = y_pred
        submission.to_csv(
            f"Classifier_{CLF.__class__.__name__}, SMOTE_{KMEANS_USE}, Train_test_split_{TRAIN_TEST_SPLIT}.csv",
            index=False,
        )
        print(f"Submission Down!")
        print(f"File name: {CLF.__class__.__name__} {KMEANS_USE} {TRAIN_TEST_SPLIT}")


if __name__ == "__main__":
    main()
