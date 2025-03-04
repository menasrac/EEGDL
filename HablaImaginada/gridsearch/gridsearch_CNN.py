import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
with open("old/scattering_features.pkl", "rb") as f:
    data_dict = pickle.load(f)

X_train = data_dict["X_train"].astype(np.float32)  # (n_samples, n_channels, n_time_steps)
y_train = data_dict["y_train"].astype(np.float32)


# Conversion en tenseurs
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Création du dataset
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.75 * len(dataset))
val_size = int(0.125 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


device = "cuda" if torch.cuda.is_available() else "cpu"

# Définition du CNN paramétrable
class CNN(nn.Module):
    def __init__(self, in_channels, conv_filters, kernel_sizes, fc_units, dropout):
        super(CNN, self).__init__()
        layers = []
        prev_channels = in_channels
        for filters, kernel_size in zip(conv_filters, kernel_sizes):
            layers.append(nn.Conv1d(prev_channels, filters, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev_channels = filters
        
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        conv_output_size = prev_channels * (X_train.shape[2] // (2 ** len(conv_filters)))
        fc_layers = []
        prev_dim = conv_output_size
        for units in fc_units:
            fc_layers.append(nn.Linear(prev_dim, units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = units
        
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

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
    return avg_loss, acc, all_preds, all_targets

# Grid Search
param_grid = {
    "conv_filters": [(16, 32), (32, 64), (64, 128)],
    "kernel_sizes": [(3, 3), (5, 5)],
    "fc_units": [(128,), (256, 128)],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.01, 0.001],
    "batch_size": [32, 64]
}

best_acc = 0.0
best_params = None

for params in itertools.product(*param_grid.values()):
    conv_filters, kernel_sizes, fc_units, dropout, lr, batch_size = params
    print(f"\nTesting config: {params}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = CNN(X_train.shape[1], conv_filters, kernel_sizes, fc_units, dropout).to(device)
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
        torch.save(model.state_dict(), "best_cnn_model.pth")
        print("New best model saved!")

print(f"\nBest accuracy: {best_acc:.4f} with params {best_params}")

# Charger le meilleur modèle et évaluer sur le test
best_model = CNN(X_train_tensor.shape[1], best_params[0], best_params[1], best_params[2], best_params[3]).to(device)
best_model.load_state_dict(torch.load("best_cnn_model.pth"))

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
plt.savefig("rendu/figures/confusion_matrix_CNN.png", format="png")

