import numpy as np
import pandas as pd
from rich import print
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import KMeansSMOTE
from utils import show_result

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CreditCardDataset


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "uci_credit_card_default.csv"
TEST_DIR = "dataset/test.csv"
TEST_SIZE = 0.3
KMEANS_USE = True
BATCH_SIZE = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
EPOCHS = 200


class LinerNet(nn.Module):
    """Linear neural network."""

    def __init__(self, in_features: int):
        super(LinerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


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

    # assign data and label
    X, y = X.values, y
    sample_size, n_features = X.shape

    # K-means SMOTE overrsampling
    if KMEANS_USE:
        k = int(np.sqrt(sample_size / 2))
        sm = KMeansSMOTE(k_neighbors=k, sampling_strategy="auto")
        X, y = sm.fit_resample(X, y)

    # Dataset
    training_dataset = CreditCardDataset(X, y)

    # DataLoader
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model
    model = LinerNet(n_features).to(DEVICE)

    # optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    # #################
    # # training loop #
    # #################

    for epoch in range(EPOCHS):
        train_accuracy = 0
        train_count = 0
        train_loss = 0.0

        for _, (x, labels) in enumerate(training_loader):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(x.float())
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(outputs.data, dim=1)
            train_count += len(x)
            train_accuracy += (y_pred == labels).sum().item()
            train_loss += loss.item() * len(labels)

        if (epoch + 1) % 10 == 0:
            print(
                f"【EPOCHS {epoch+1}】: Train Loss: {(train_loss / train_count):.3f}, Train ACC: {(train_accuracy / train_count):.3f}"
            )

    # Create submission
    X_test = torch.tensor(test.values)
    with torch.no_grad():
        X_test = X_test.to(DEVICE)
        outputs = model(X_test.float())
        y_pred = torch.argmax(outputs.data, dim=1)
        y_pred = ["Y" if y == 1 else "N" for y in y_pred]
        SUBMISSION = "dataset/sample_submission.csv"
        submission = pd.read_csv(SUBMISSION)
        submission["PAY"] = y_pred
        submission.to_csv("Submission.csv", index=False)
        print(f"Submission Down!")


if __name__ == "__main__":
    main()
