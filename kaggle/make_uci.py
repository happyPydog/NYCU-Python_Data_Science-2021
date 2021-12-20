import sys
from collections import Counter
import numpy as np
import pandas as pd
from rich import print
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import KMeansSMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CreditCardDataset


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset/UCI_Credit_Card.csv"
RANDOM_SEED = 98765
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
            nn.Linear(in_features, 55),
            nn.ReLU(),
            nn.Linear(55, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


def main():

    # load data and drop some unuseless features
    df = pd.read_csv(DATA_DIR).drop(
        columns=[
            "ID",
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
            df["AGE"],
            df["SEX"],
            df["EDU"],
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

    df.to_csv("uci_credit_card_default.csv", index=False)
