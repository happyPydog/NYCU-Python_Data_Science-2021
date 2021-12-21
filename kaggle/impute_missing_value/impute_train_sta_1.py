import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import LinerNet
from preprocessing import min_max_scaler
from dataset import CreditCardDataset
from utils import RunTime

# device of torch.cuda
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# config of training model
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
EPOCHS = 2000


@RunTime()
def main():

    # load csv
    df = pd.read_csv("dataset/UCI_Credit_Card.csv")

    # drop unuseless feautes in training
    df = df.drop(columns=["default.payment.next.month"])

    # split data to train and test
    df = df.drop(
        columns=[
            "ID",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]
    )

    # rename
    df = df.rename(columns={"PAY_0": "PAY_1"})

    # reorder df
    df = pd.concat(
        [
            df["AGE"],
            df["SEX"],
            df["EDUCATION"],
            df["MARRIAGE"],
            df["BILL_AMT1"],
            df["BILL_AMT2"],
            df["BILL_AMT3"],
            df["PAY_1"],
            df["PAY_2"],
            df["PAY_3"],
            df["PAY_AMT1"],
            df["PAY_AMT2"],
            df["PAY_AMT3"],
        ],
        axis=1,
    )

    # Assign data and label
    X, y = df.drop(columns="PAY_1"), df["PAY_1"]

    print(X)
    print(y)

    import sys

    sys.exit()

    # test drop STA_1
    test = test.drop(columns="STA_1")
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # label encoder
    le = LabelEncoder()
    le.fit(train["STA_1"])
    train["STA_1"] = le.transform(train["STA_1"])

    # onehat encoder

    # scaling
    train = min_max_scaler(train)
    test = min_max_scaler(test)

    # assign X, y
    X, y = train.drop(columns="STA_1").values, train["STA_1"].values
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

        if (epoch + 1) % 10 == 0:
            print(
                f"【EPOCHS {epoch+1}】: Train Loss: {(train_loss / train_count):.3f}, Train ACC: {(train_accuracy / train_count):.3f}, Test Loss: {(test_loss / test_count):.3f}, Test ACC: {(test_accuracy / test_count):.3f}"
            )

    # save model
    torch.save(model, "model_sta_1.pt")

    # save result
    model = torch.load("model_sta_1.pt")
    test = torch.tensor(test.values)

    model.eval()
    with torch.no_grad():
        outputs = model(test.float().to(DEVICE))
        y_pred = torch.argmax(outputs, dim=1).to("cpu")
        sta_1 = le.inverse_transform(y_pred)
        df["STA_1"][df["STA_1"].isna()] = sta_1
        df.to_csv("STA_1.csv", index=False)


if __name__ == "__main__":
    main()
