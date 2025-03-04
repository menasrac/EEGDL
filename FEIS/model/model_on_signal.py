import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
from train_model import *
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from scipy import signal


NB_EPOCH = 200
BATCH_SIZE = 16

DATA_PATH = "./experiments"

data = []

fs = 256  # Sampling frequency (Hz)
nyquist = fs/2
low = 10/nyquist
high = 100/nyquist
b, a = signal.butter(4, [low, high], btype='bandpass')

for subject in [str(i).zfill(2) for i in range(1, 21)]:
    df_sub = pd.read_csv(f"{DATA_PATH}/{subject}/thinking.csv")
    df_sub = df_sub.groupby("Epoch")
    for block_id, block in df_sub:
        label = block["Label"].iloc[0]
        eeg_data = block.iloc[:, 2:16].values
        normalised = (eeg_data - eeg_data.mean(axis=0)) / (eeg_data.std(axis=0) + 1e-8)

        # Apply filter to each channel
        #banded = signal.filtfilt(b, a, normalised, axis=0)

        data.append({"Features": normalised, "Label": label})

data = pd.DataFrame(data)

print(data.head())
print(data.info())

# Split data in train, val, test
train_dataset = data.sample(frac=0.70)
val_dataset = data.drop(train_dataset.index)
test_dataset = val_dataset.sample(frac=0.50)
val_dataset = val_dataset.drop(test_dataset.index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_sub = pd.read_csv(f"experiments/01/thinking.csv")
labels = df_sub["Label"].unique()
label_mapping = {label: i for i, label in enumerate(labels)}

class EEGDataset(Dataset):
    def __init__(self, dataframe, label_mapping):
        self.X = torch.stack([torch.tensor(x.copy(), dtype=torch.float32) for x in dataframe["Features"]])
        # Créer un mapping des labels uniques en entiers
        print("Label mapping:", label_mapping)
        # Convertir les labels en indices
        self.y = torch.tensor([label_mapping[label] for label in dataframe["Label"]], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        return features, label

# Create the custom datasets
train_dataset = EEGDataset(train_dataset,label_mapping)
val_dataset = EEGDataset(val_dataset,label_mapping)
test_dataset = EEGDataset(test_dataset,label_mapping)


# Créer les DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



class LSTMSignal(nn.Module):
    def __init__(self):
        super(LSTMSignal, self).__init__()
        self.lstm = nn.LSTM(14, 64, num_layers=5, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # On récupère la dernière sortie temporelle
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Instanciation du modèle
model = LSTMSignal().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0001)

criterion = nn.CrossEntropyLoss()

# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = None

# Lancer l'entraînement
train_model(model, train_loader, val_loader, test_loader, NB_EPOCH, device, optimizer, criterion, scheduler)
