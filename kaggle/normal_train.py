import numpy as np
import pandas as pd
import xgboost as xgb
from rich import print
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import KMeansSMOTE


UCI_DIR = "./uci.csv"
TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"
TEST_SIZE = 0.05
TRAIN_TEST_SPLIT = True
KMEANS_USE = False
USE_CUI = False

CLF = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    max_depth=6,
    gamma=0.1,
    scale_pos_weight=0.9,
    learning_rate=0.1,
    reg_lambda=0.01,
    min_child_weight=1.2,
    alpha=0.11,
    max_delta_step=6,
    subsample=0.9,
)


def main():

    # load dataset
    train = pd.read_csv(TRAIN_DIR).drop(columns=["ID"])
    test = pd.read_csv(TEST_DIR).drop(columns=["ID"])
    uci = pd.read_csv(UCI_DIR).drop(columns=["ID"])

    # label encoder for the target value
    X, label = train.drop(columns="PAY"), train["PAY"].values
    X_UCI, Y_CUI = uci.drop(columns="PAY"), uci["PAY"].values
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(label)
    le.fit(Y_CUI)
    y_uci = le.transform(Y_CUI)

    # impute missing values for testing data
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_most_frequent.fit(X.values)
    X = pd.DataFrame(imp_most_frequent.transform(X.values), columns=X.columns)
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
    X = X.astype(col_numerical)
    test = test.astype(col_numerical)

    # onehotencoder
    df_dummy = pd.get_dummies(X)
    X = pd.DataFrame(df_dummy)
    df_dummy = pd.get_dummies(test)
    test = pd.DataFrame(df_dummy)
    df_dummy = pd.get_dummies(X_UCI)
    X_UCI = pd.DataFrame(df_dummy)

    # normalize
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    test = pd.DataFrame(scaler.transform(test))
    scaler.fit(X_UCI)
    X_UCI = pd.DataFrame(scaler.transform(X_UCI))

    # assign data and label
    X, y = X.values, y
    X_test = test.values
    X_uci = X_UCI.values
    sample_size, _ = X.shape

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, stratify=y, random_state=100
    )
    print(f"{X_train.shape= }, {y_train.shape= }, {X_val.shape= }, {y_val.shape= }")
    if USE_CUI:
        X_train, X_val, y_train, y_val = train_test_split(
            X_uci,
            y_uci,
            test_size=TEST_SIZE,
            shuffle=True,
            stratify=y_uci,
            random_state=69,
        )
        print(f"{X_train.shape= }, {y_train.shape= }, {X_val.shape= }, {y_val.shape= }")

    # K-means SMOTE overrsampling
    if KMEANS_USE:
        k = int(np.sqrt(sample_size / 2))
        sm = KMeansSMOTE(
            k_neighbors=k, cluster_balance_threshold="auto", sampling_strategy="auto"
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
        f"UCI_{USE_CUI}, Classifier_{CLF.__class__.__name__}, SMOTE_{KMEANS_USE}, Train_test_split_{TRAIN_TEST_SPLIT}.csv",
        index=False,
    )
    print(f"Submission Down!")
    print(
        f"File name: uci_dataset {USE_CUI} {CLF.__class__.__name__} {KMEANS_USE} {TRAIN_TEST_SPLIT}"
    )


if __name__ == "__main__":
    main()
