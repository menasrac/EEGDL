import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


# Charger les données
with open("./swt_features_9sujets.pkl", "rb") as f:
    data_dict = pickle.load(f)

# Les données sont supposées être de forme (n_samples, n_channels, n_time_steps)
X_train = data_dict["X_train"].astype(np.float32)
y_train = data_dict["y_train"].astype(np.float32)

# Pour le LSTM, on souhaite une entrée de forme (n_samples, time_steps, features)
# Ici, on transpose de (n_samples, n_channels, n_time_steps) à (n_samples, n_time_steps, n_channels)
X_train = np.transpose(X_train, (0, 2, 1))
print("Forme finale des données pour LSTM :", X_train.shape)

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Création du dataset et découpage en ensembles train/validation/test
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.75 * len(dataset))
val_size = int(0.125 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Définition du modèle LSTM paramétrable
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_units, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        # Construction des couches fully-connected
        fc_layers = []
        prev_dim = hidden_size
        for units in fc_units:
            fc_layers.append(nn.Linear(prev_dim, units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = units
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x de forme (batch, time_steps, features)
        out, _ = self.lstm(x)
        # Utiliser la sortie du dernier pas de temps
        out = out[:, -1, :]
        return self.fc(out)

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

# Grid Search des hyperparamètres pour le LSTM
param_grid = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "fc_units": [(128,), (256, 128)],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.01, 0.001],
    "batch_size": [16, 32, 64]
}

best_acc = 0.0
best_params = None

for params in itertools.product(*param_grid.values()):
    hidden_size, num_layers, fc_units, dropout, lr, batch_size = params
    print(f"\nTesting config: {params}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = LSTMClassifier(input_size=X_train.shape[2],
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           fc_units=fc_units,
                           dropout=dropout).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    patience = 5
    wait = 0
    
    for epoch in range(50):  # Early stopping
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
        print(f"Epoch {epoch+1:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
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
        torch.save(model.state_dict(), "best_lstm_model.pth")
        print("New best model saved!")

print(f"\nBest accuracy: {best_acc:.4f} with params {best_params}")

# Charger le meilleur modèle et évaluer sur le set de test
best_model = LSTMClassifier(input_size=X_train_tensor.shape[2],
                            hidden_size=best_params[0],
                            num_layers=best_params[1],
                            fc_units=best_params[2],
                            dropout=best_params[3]).to(device)
best_model.load_state_dict(torch.load("best_lstm_model.pth"))

test_loader = DataLoader(test_dataset, batch_size=32)
test_loss, test_acc, all_preds, all_targets = evaluate(best_model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

# Calcul et affichage de la matrice de confusion
conf_matrix = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.savefig("./figures/confusion_matrix_LSTM.png", format="png")
