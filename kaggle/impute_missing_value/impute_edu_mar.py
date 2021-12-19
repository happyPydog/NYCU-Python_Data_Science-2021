import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"


if __name__ == "__main__":

    df = pd.read_csv(TRAIN_DIR)
    df_test = pd.read_csv(TEST_DIR)

    # EDU 和 MAR 這兩類的缺失值使用"頻率最高"的那一類來補缺失值
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    edu = df["EDU"]
    mar = df["MAR"]
    imp_most_frequent.fit(edu.values.reshape(-1, 1))
    df["EDU"] = pd.DataFrame(imp_most_frequent.transform(edu.values.reshape(-1, 1)))
    imp_most_frequent.fit(mar.values.reshape(-1, 1))
    df["MAR"] = pd.DataFrame(imp_most_frequent.transform(mar.values.reshape(-1, 1)))
    df.to_csv("dataset/train_IMV_EDU_MAR.csv", index=False)
