import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import LinerNet
from preprocessing import min_max_scaler, standard_scaler
from dataset import CreditCardDataset
from utils import RunTime

# device of torch.cuda
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# config of training model
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 2000


@RunTime()
def main():

    # load csv
    train_csv = pd.read_csv("dataset/train.csv")
    test_csv = pd.read_csv("dataset/test.csv")

    # concat train.csv and test.csv as a new df
    df = pd.concat([train_csv, test_csv])

    # drop unuseless feautes in training
    df = df.drop(columns=["AGE", "SEX", "EDU", "MAR", "PAY"])

    # split data to train and test
    train = df.drop(columns=["ID", "STA_1", "STA_3"]).dropna()
    test = df.drop(columns=["ID", "STA_1", "STA_3"])[
        df.drop(columns=["STA_1", "STA_3"]).STA_2.isnull()
    ]

    # test drop STA_1
    test = test.drop(columns="STA_2")
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # label encoder
    le = LabelEncoder()
    le.fit(train["STA_2"])
    train["STA_2"] = le.transform(train["STA_2"])

    # scaling
    train = min_max_scaler(train)
    test = min_max_scaler(test)

    # assign X, y
    X, y = train.drop(columns="STA_2").values, train["STA_2"].values
    _, n_features = X.shape
    print(f"{X.shape}, {y.shape}")
    print(test.values.shape)

    # Dataset
    training_dataset = CreditCardDataset(X, y)

    # DataLoader
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model
    model = LinerNet(n_features).to(DEVICE)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    #################
    # training loop #
    #################

    for epoch in range(EPOCHS):
        accuracy = 0
        count = 0
        training_loss = 0.0
        for _, (x, labels) in enumerate(training_loader):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(x.float())
            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(outputs.data, dim=1)
            count += len(x)
            accuracy += (y_pred == labels).sum().item()
            training_loss += loss.item() * len(labels)

        if (epoch + 1) % 10 == 0:
            print(
                f"【EPOCHS {epoch+1}】: Train Loss: {(training_loss / count):.3f}, Train ACC: {(accuracy / count):.3f}"
            )

    # save model
    torch.save(model, "model_sta_2.pt")

    # save result
    model = torch.load("model_sta_2.pt")
    test = torch.tensor(test.values)

    model.eval()
    with torch.no_grad():
        outputs = model(test.float().to(DEVICE))
        y_pred = torch.argmax(outputs, dim=1).to("cpu")
        sta_1 = le.inverse_transform(y_pred)
        df["STA_2"][df["STA_2"].isna()] = sta_1
        df.to_csv("STA_2.csv", index=False)


if __name__ == "__main__":
    main()
