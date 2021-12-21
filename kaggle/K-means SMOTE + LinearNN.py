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
TRAIN_DIR = "dataset/train_imputed.csv"
TEST_DIR = "dataset/test_imputed.csv"
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

    # load data
    train = pd.read_csv(TRAIN_DIR).drop(columns=["ID"])
    test = pd.read_csv(TEST_DIR).drop(columns=["ID"])

    # impute missing value
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_most_frequent.fit(train.values)
    train = pd.DataFrame(
        imp_most_frequent.transform(train.values), columns=train.columns
    )
    imp_most_frequent.fit(test.values)
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
    train = train.astype(col_numerical)
    test = test.astype(col_numerical)

    # onehot-encoder
    train_pay = train["PAY"]
    df_dummy = pd.get_dummies(train.drop(columns="PAY"))
    train = pd.DataFrame(df_dummy)
    df_dummy = pd.get_dummies(test)
    test = pd.DataFrame(df_dummy)

    # pd.DataFrame to array
    X, y = train.values, train_pay.values
    sample_size, n_features = X.shape

    # labelencoder for target value
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Original: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}")

    # Min-Max nomralize
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # PCA

    # K-means SMOTE
    k = int(np.sqrt(sample_size / 2))
    sm = KMeansSMOTE(
        k_neighbors=k, sampling_strategy=0.8, cluster_balance_threshold="auto"
    )
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train).items()))
    print(
        f"After oversamping: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}"
    )

    # Dataset
    training_dataset = CreditCardDataset(X_train, y_train)
    val_dataset = CreditCardDataset(X_val, y_val)

    # DataLoader
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        test_accuracy = 0
        test_count = 0
        test_loss = 0.0

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

        with torch.no_grad():
            for _, (x, labels) in enumerate(val_loader):
                x = x.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(x.float())
                loss = criterion(outputs, labels.long())
                y_pred = torch.argmax(outputs.data, dim=1)
                test_count += len(x)
                test_accuracy += (y_pred == labels).sum().item()
                test_loss += loss.item() * len(labels)

        if (epoch + 1) % 10 == 0:
            print(
                f"【EPOCHS {epoch+1}】: Train Loss: {(train_loss / train_count):.3f}, Train ACC: {(train_accuracy / train_count):.3f}, Test Loss: {(test_loss / test_count):.3f}, Test ACC: {(test_accuracy / test_count):.3f}"
            )

    # # save model
    # torch.save(model, "K-means SMOTE + Linear.pt")

    # # save result
    # model = torch.load("model_sta_1.pt")
    # test = torch.tensor(test.values)

    # model.eval()
    # with torch.no_grad():
    #     outputs = model(test.float().to(DEVICE))
    #     y_pred = torch.argmax(outputs, dim=1).to("cpu")
    #     sta_1 = le.inverse_transform(y_pred)
    #     df["STA_1"][df["STA_1"].isna()] = sta_1
    #     df.to_csv("STA_1.csv", index=False)


if __name__ == "__main__":
    main()
