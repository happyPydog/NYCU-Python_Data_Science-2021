import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


def min_max_scaler(df: pd.DataFrame):

    column_trans = ColumnTransformer(
        [
            (
                "scale",
                MinMaxScaler(),
                ["CRE", "BILL_1", "BILL_2", "BILL_3", "AMT_1", "AMT_2", "AMT_3"],
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_trans.fit(df)
    for idx, col in enumerate(column_trans.get_feature_names_out()):
        df[col] = column_trans.transform(df)[:, idx]

    return df


def standard_scaler(df: pd.DataFrame):

    column_trans = ColumnTransformer(
        [
            (
                "scale",
                StandardScaler(),
                ["CRE", "BILL_1", "BILL_2", "BILL_3", "AMT_1", "AMT_2", "AMT_3"],
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_trans.fit(df)
    for idx, col in enumerate(column_trans.get_feature_names_out()):
        df[col] = column_trans.transform(df)[:, idx]

    return df
