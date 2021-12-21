import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from model import LinerNet
from dataset import CreditCardDataset
from utils import RunTime

# device of torch.cuda
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# config of training model
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
EPOCHS = 10000


def min_max_scaler(df: pd.DataFrame):

    column_trans = ColumnTransformer(
        [
            (
                "scale",
                MinMaxScaler(),
                ["CRE", "BILL_3", "AMT_3"],
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_trans.fit(df)
    for idx, col in enumerate(column_trans.get_feature_names_out()):
        df[col] = column_trans.transform(df)[:, idx]

    return df


@RunTime()
def main():

    # load csv
    train_csv = pd.read_csv("dataset/train.csv")
    test_csv = pd.read_csv("dataset/test.csv")

    # concat train.csv and test.csv as a new df
    df = pd.concat([train_csv, test_csv])

    # drop unuseless feautes in training
    df = df.drop(columns=["PAY"])

    # split data to train and test
    train = df.drop(
        columns=["ID", "BILL_1", "BILL_2", "AMT_1", "AMT_2", "STA_1", "STA_2"]
    ).dropna()
    test = df.drop(
        columns=["ID", "BILL_1", "BILL_2", "AMT_1", "AMT_2", "STA_1", "STA_2"]
    )[df.drop(columns=["STA_1", "STA_2"]).STA_3.isnull()]

    # test drop STA_1
    test = test.drop(columns="STA_3")
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # label encoder
    le = LabelEncoder()
    le.fit(train["STA_3"])
    train["STA_3"] = le.transform(train["STA_3"])

    # onehot encoder
    y = train["STA_3"].values
    df_dummy = pd.get_dummies(train)
    train = pd.DataFrame(df_dummy)

    df_dummy = pd.get_dummies(test)
    test = pd.DataFrame(df_dummy)

    # scaling
    train = min_max_scaler(train)
    test = min_max_scaler(test)

    # assign X, y
    X, y = train.drop(columns="STA_3").values, train["STA_3"].values
    _, n_features = X.shape
    print(f"{X.shape}, {y.shape}")
    print(test.values.shape)

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    print(f"Original: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}")

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

    #################
    # training loop #
    #################

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

        if (epoch + 1) % 100 == 0:
            print(
                f"【EPOCHS {epoch+1}】: Train Loss: {(train_loss / train_count):.3f}, Train ACC: {(train_accuracy / train_count):.3f}, Test Loss: {(test_loss / test_count):.3f}, Test ACC: {(test_accuracy / test_count):.3f}"
            )

    # save model
    torch.save(model, "model_sta_3.pt")

    # save result
    model = torch.load("model_sta_3.pt")
    test = torch.tensor(test.values)

    model.eval()
    with torch.no_grad():
        outputs = model(test.float().to(DEVICE))
        y_pred = torch.argmax(outputs, dim=1).to("cpu")
        sta_1 = le.inverse_transform(y_pred)
        df["STA_3"][df["STA_3"].isna()] = sta_1
        df.to_csv("STA_3.csv", index=False)


if __name__ == "__main__":
    main()
