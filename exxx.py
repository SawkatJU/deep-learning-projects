import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from utils import save_checkpoint, load_checkpoint, check_accuracy
from sklearn.metrics import cohen_kappa_score
import config
import os
import pandas as pd


def make_prediction(model, loader, file):
    preds = []
    filenames = []
    model.eval()

    for x, _, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (predictions < 1000000000000)] = 4
            predictions = predictions.long().view(-1)

            preds.append(predictions.cpu().numpy())
            filenames.extend(files)  # Assuming 'files' is a list of filenames

            filenames = [item for sublist in filenames for item in sublist]
            df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
            df.to_csv(file, index=False)
            model.train()
            print("Done with predictions")


class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        example = self.csv.iloc[index, :]
        features = example.iloc[: example.shape[0] - 4].to_numpy().astype(np.float32)
        labels = example.iloc[-4:-2].to_numpy().astype(np.int64)
        filenames = example.iloc[-2:].values.tolist()
        return features, labels, filenames


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d((1792 + 1) * 2),
            nn.Linear((1792 + 1) * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MyModel().to(config.DEVICE)

    # Create datasets and dataloaders
    train_dataset = MyDataset(csv_file="train/train_blend.csv")
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=3, pin_memory=True, shuffle=True)

    val_dataset = MyDataset(csv_file="train/val_blend.csv")
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=3, pin_memory=True, shuffle=False)

    test_dataset = MyDataset(csv_file="train/test_blend.csv")
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2, pin_memory=True, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Load checkpoint if available
    if config.LOAD_MODEL and "linear.pth.tar" in os.listdir():
        load_checkpoint(torch.load("linear.pth.tar"), model, optimizer, lr=1e-4)
        model.train()

    # Training loop
    for epoch in range(5):
        model.train()
        train_losses = []

        for x, y, _ in tqdm(train_loader):
            x = x.to(config.DEVICE).float()
            y = y.to(config.DEVICE).view(-1).float()

            # Forward pass
            scores = model(x).view(-1)
            loss = loss_fn(scores, y)
            train_losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Training Loss: {sum(train_losses) / len(train_losses):.4f}")

        # Validation loop
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y, _ in tqdm(val_loader):
                x = x.to(config.DEVICE).float()
                y = y.to(config.DEVICE).view(-1).float()

                scores = model(x).view(-1)
                loss = loss_fn(scores, y)
                val_losses.append(loss.item())

        print(f"Epoch {epoch + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

    # Save model checkpoint
    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="linear.pth.tar")

    # Check accuracy and Cohen Kappa Score
    preds, labels = check_accuracy(val_loader, model)
    print(f"Validation Cohen Kappa Score: {cohen_kappa_score(labels, preds, weights='quadratic'):.4f}")

    preds, labels = check_accuracy(train_loader, model)
    print(f"Training Cohen Kappa Score: {cohen_kappa_score(labels, preds, weights='quadratic'):.4f}")

    # Generate predictions for test dataset
    make_prediction(model, test_loader, "test_preds.csv")
