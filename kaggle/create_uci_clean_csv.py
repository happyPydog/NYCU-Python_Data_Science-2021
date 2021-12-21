import pandas as pd


DATA_DIR = "dataset/UCI_Credit_Card.csv"


def main():

    # load data and drop some unuseless features
    df = pd.read_csv(DATA_DIR).drop(
        columns=[
            "PAY_4",
            "PAY_5",
            "PAY_5",
            "PAY_6",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]
    )

    # rename columns
    df = df.rename(
        columns={
            "PAY_0": "STA_1",
            "PAY_2": "STA_2",
            "PAY_3": "STA_3",
            "LIMIT_BAL": "CRE",
            "EDUCATION": "EDU",
            "MARRIAGE": "MAR",
            "BILL_AMT1": "BILL_1",
            "BILL_AMT2": "BILL_2",
            "BILL_AMT3": "BILL_3",
            "PAY_AMT1": "AMT_1",
            "PAY_AMT2": "AMT_2",
            "PAY_AMT3": "AMT_3",
            "default.payment.next.month": "PAY",
        }
    )

    # reorder df
    df = pd.concat(
        [
            df["ID"],
            df["AGE"],
            df["SEX"],
            df["EDU"],
            df["MAR"],
            df["CRE"],
            df["BILL_1"],
            df["BILL_2"],
            df["BILL_3"],
            df["STA_1"],
            df["STA_2"],
            df["STA_3"],
            df["AMT_1"],
            df["AMT_2"],
            df["AMT_3"],
            df["PAY"],
        ],
        axis=1,
    )

    # classification PAY1 ~ PAY3
    for sta in ["STA_1", "STA_2", "STA_3"]:
        df[sta] = df[sta].apply(lambda x: "duly" if x <= 0 else "delay")

    # SEX to 1, 2 to "Male" and "Female"
    df["SEX"] = df["SEX"].apply(lambda x: "Male" if x == 1 else "Famale")

    # EDU 1, 2, 3 to "HighSchool", "College", "Graduate"

    def transform_edu(x):
        if x == 1:
            return "Graduate"
        elif x == 2:
            return "College"
        elif x == 3:
            return "HighSchool"
        else:
            return "Others"

    df["EDU"] = df["EDU"].apply(transform_edu)

    def transform_mar(x):
        if x == 1:
            return "married"
        elif x == 2:
            return "single"
        elif x == 3:
            return "others"

    df["MAR"] = df["MAR"].apply(transform_mar)

    # PAY to "Y" and "N"
    df["PAY"] = df["PAY"].apply(lambda x: "N" if x == 1 else "Y")

    # save csv
    df.to_csv("uci_credit_card_default.csv", index=False)


if __name__ == "__main__":
    main()
