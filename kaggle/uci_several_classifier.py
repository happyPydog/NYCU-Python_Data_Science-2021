import numpy as np
import pandas as pd
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


def main():

    # load dataset
    train = pd.read_csv(TRAIN_DIR).drop(columns=["ID"])
    test = pd.read_csv(TEST_DIR).drop(columns=["ID"])

    # impute missing values for testing data
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_most_frequent.fit(test.values)
    test = pd.DataFrame(imp_most_frequent.transform(test.values), columns=test.columns)
    test.head()

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
    test = test.astype(col_numerical)

    # print(
    #     f'STA_1 duly: {len(test[test["STA_1"] == "duly"]) / len(test)}, STA_1 delay: {len(test[test["STA_1"] == "delay"]) / len(test)}'
    # )
    # print(
    #     f'STA_2 duly: {len(test[test["STA_2"] == "duly"]) / len(test)}, STA_2 delay: {len(test[test["STA_2"] == "delay"]) / len(test)}'
    # )
    # print(
    #     f'STA_3 duly: {len(test[test["STA_3"] == "duly"]) / len(test)}, STA_3 delay: {len(test[test["STA_3"] == "delay"]) / len(test)}'
    # )

    # OneHot-encoder
    X, y = train.drop(columns="PAY"), train.PAY.values
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    df_dummy = pd.get_dummies(X)
    X = pd.DataFrame(df_dummy)
    df_dummy = pd.get_dummies(test)
    test = pd.DataFrame(df_dummy)
    cols = X.columns

    # Min-Max Normalize
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=cols)
    scaler.fit(test)
    test = pd.DataFrame(scaler.transform(test), columns=cols)

    # tran test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, shuffle=True, test_size=TEST_SIZE, stratify=y
    )
    sample_size, n_features = X_train.shape

    # K-means SMOTE overrsampling
    if KMEANS_USE:
        k = int(np.sqrt(sample_size / 2))
        sm = KMeansSMOTE(k_neighbors=k, sampling_strategy="auto")
        X_train, y_train = sm.fit_resample(X_train, y_train)


if __name__ == "__main__":
    main()
