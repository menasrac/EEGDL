import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données SWT
with open("./swt_features_5sujets.pkl", "rb") as f:
    data_dict = pickle.load(f)

X_train = data_dict["X_train"]  # Forme : (470, 3072, 10)
y_train = data_dict["y_train"].astype(np.float32)

print("Forme initiale des données d'entraînement :", X_train.shape)

# Vérification de la distribution des classes
unique, counts = np.unique(y_train, return_counts=True)
print("Distribution des classes:", dict(zip(unique, counts)))

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Création du dataset et découpage en ensembles d'entraînement et de validation
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Définition de l'architecture LSTM pour la classification binaire
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, fc_units=(64,), dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        # Construction de la ou des couches fully-connected
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
        # x a la forme (batch, time_steps, input_size)
        out, _ = self.lstm(x)
        # Utiliser la sortie du dernier pas de temps
        out = out[:, -1, :]
        return self.fc(out)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier(input_size=X_train.shape[2]).to(device)

# Définition de la loss et de l'optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, loader, device):
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
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, all_preds, all_targets

num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
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
    val_loss, val_preds, val_targets = evaluate(model, val_loader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(classification_report(val_targets, val_preds, digits=4))

torch.save(model.state_dict(), "lstm_model_swt.pth")
print("Modèle sauvegardé dans lstm_model_swt.pth")

# Courbes d'apprentissage
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.legend()
plt.title("Courbes d'apprentissage")
plt.savefig("learning_curves_lstm_swt.png", format="png")

# Matrice de confusion sur le set de validation
_, _, val_targets = evaluate(model, val_loader, device)
_, val_preds, _ = evaluate(model, val_loader, device)
conf_matrix = confusion_matrix(val_targets, val_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.savefig("confusion_matrix_lstm_swt.png", format="png")
