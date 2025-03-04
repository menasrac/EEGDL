import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
with open("old/dwt_features.pkl", "rb") as f:
    data_dict = pickle.load(f)

X_train = data_dict["X_train"].astype(np.float32)  # (n_samples, n_channels, n_time_steps)
y_train = data_dict["y_train"].astype(np.float32)

# Conversion en tenseurs
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], -1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Création du dataset
dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Séparer en train (75%), val (12,5%) et test (12,5%)
train_size = int(0.75 * len(dataset))
val_size = int(0.125 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Définition du MLP paramétrable
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Fonction d'évaluation
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, all_preds, all_targets  # Ajout des prédictions et des cibles


# Grid Search
param_grid = {
    "hidden_layers": [(256, 128), (512, 256, 128), (512, 256, 128, 64)],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [0.01, 0.001, 0.0005],
    "batch_size": [16, 32, 64],
}

best_acc = 0.0
best_params = None

for params in itertools.product(*param_grid.values()):
    hidden_layers, dropout, lr, batch_size = params
    print(f"\nTesting config: {params}")
    
    # Chargement des loaders avec les données train et val
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = MLP(X_train_tensor.shape[1], hidden_layers, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    patience = 5
    wait = 0
    
    for epoch in range(50):  # Early Stopping
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = params
        torch.save(model.state_dict(), "best_mlp_model.pth")
        print("New best model saved!")

print(f"\nBest accuracy: {best_acc:.4f} with params {best_params}")

# Charger le meilleur modèle et évaluer sur le test
best_model = MLP(X_train_tensor.shape[1], best_params[0], best_params[1]).to(device)
best_model.load_state_dict(torch.load("best_mlp_model.pth"))

# Création du loader test
test_loader = DataLoader(test_dataset, batch_size=32)

# Evaluation sur le set de test
test_loss, test_acc, all_preds, all_targets = evaluate(best_model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(all_targets, all_preds)

# Affichage et sauvegarde de la matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.savefig("rendu/figures/confusion_matrix_MLP_dwt.png", format="png")
