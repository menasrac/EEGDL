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


NB_EPOCH = 200
BATCH_SIZE = 16

DATA_PATH = "./preprocessing"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_sub = pd.read_csv(f"experiments/01/thinking.csv")
labels = df_sub["Label"].unique()
label_mapping = {label: i for i, label in enumerate(labels)}

class EEGDataset(Dataset):
    def __init__(self, features, labels, label_mapping):
        self.X = features
        print("Label mapping:", label_mapping)
        self.y = [label_mapping[label] for label in labels]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_array = np.array(self.X[idx], dtype=np.float32)
        x_array = np.transpose(x_array, (1, 0))  # Adjusted to match the 2D array
        x_array = x_array.reshape(x_array.shape[0], -1)
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor



# CHARGER LES DONNÉES 
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Charger les données d'entraînement et de test
train_data = load_data(DATA_PATH + "/train_scattering.pkl")
test_data = load_data(DATA_PATH + "/test_scattering.pkl")

# Affichage de la tête des données pour vérification
print(train_data.head())
print(test_data.loc[0]["Features"].shape)

# Séparer les entrées et les cibles pour les deux ensembles
train_inputs, train_targets = train_data['Features'], train_data['Label']
print(train_inputs.shape)
test_inputs, test_targets = test_data['Features'], test_data['Label']

# Créer des datasets personnalisés avec la classe EEGDataset
train_dataset = EEGDataset(train_inputs, train_targets, label_mapping)
test_dataset = EEGDataset(test_inputs, test_targets, label_mapping)

# Créer un ensemble de validation à partir de l'ensemble d'entraînement
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# Créer les DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



class FEIS_LSTM(nn.Module):
    def __init__(self):
        super(FEIS_LSTM, self).__init__()
        # Remplacer 14 par 1764 pour correspondre à la taille de l'entrée aplatie, jsp si c'est la bonne idée
        self.lstm = nn.LSTM(1764, 128, num_layers=7, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        x = hidden[-1]  # On récupère la dernière sortie temporelle
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Instanciation du modèle
model = FEIS_LSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
scheduler = None

# Lancer l'entraînement
train_model(model, train_loader, val_loader, test_loader, NB_EPOCH, device, optimizer, criterion, scheduler)
