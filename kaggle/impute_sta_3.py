from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import KMeansSMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CreditCardDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/train.csv"
TEST_DIR = "dataset/test.csv"
UCI_DIR = "uci_credit_card_default.csv"
EPOCHS = 100


class LinerNet(nn.Module):
    """Linear neural network."""

    def __init__(self, in_features: int):
        super(LinerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


def main():

    df = pd.read_csv(TEST_DIR)

    # load uci_data
    df_uci = pd.read_csv(UCI_DIR).loc[:, ("BILL_3", "AMT_3", "STA_3")]
    df_test = pd.read_csv(TEST_DIR).loc[:, ("BILL_3", "AMT_3")]

    # label encoder for the target value
    le = LabelEncoder()
    le.fit(df_uci["STA_3"].values)
    label = le.transform(df_uci["STA_3"].values)

    # Min-Max normalize
    X, y = df_uci.drop(columns="STA_3").values, label
    sample_size, n_features = X.shape
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(df_test.values)
    print(f"{X.shape = }, {X_test.shape = }")

    # Dataset
    training_dataset = CreditCardDataset(X, y)

    # DataLoader
    training_loader = DataLoader(training_dataset, batch_size=200, shuffle=True)

    # model
    model = LinerNet(n_features).to(DEVICE)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    #################
    # training loop #
    #################

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

    # save model
    torch.save(model, "model_sta_3.pt")

    # save result
    model = torch.load("model_sta_3.pt")
    X_test = torch.tensor(X_test)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test.float().to(DEVICE))
        y_pred = torch.argmax(outputs, dim=1).to("cpu")
        sta_3 = le.inverse_transform(y_pred)
        df["STA_3"][df["STA_3"].isna()] = sta_3
        df.to_csv("STA_3.csv", index=False)


if __name__ == "__main__":
    main()
