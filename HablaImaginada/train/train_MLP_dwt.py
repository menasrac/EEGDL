import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données à partir du pickle
with open("old/dwt_features.pkl", "rb") as f:
    data_dict = pickle.load(f)

X_train = data_dict["X_train"]  # shape (n_samples, n_channels, n_time_steps)
X_test = data_dict["X_test"]
y_train = data_dict["y_train"].astype(np.float32)

print("Forme des données d'entraînement :", X_train.shape)

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Aplatir les dimensions restantes
X_train_tensor = X_train_tensor.view(X_train_tensor.shape[0], -1)

# Création d'un dataset et découpage en train/validation
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Définition de l'architecture MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Déterminer la dimension d'entrée
input_dim = X_train_tensor.shape[1]
model = MLP(input_dim=input_dim).to("cuda" if torch.cuda.is_available() else "cpu")

# Définition de la fonction de perte et de l'optimiseur
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
            
            # Aplatir les dimensions restantes
            X_batch = X_batch.view(X_batch.size(0), -1)

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

# Entraînement du modèle
num_epochs = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Aplatir les dimensions restantes
        X_batch = X_batch.view(X_batch.size(0), -1)
        
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
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    print(classification_report(val_targets, val_preds, digits=4))

# Sauvegarde du modèle
torch.save(model.state_dict(), "mlp_model.pth")
print("Modèle sauvegardé dans mlp_model.pth")

# Affichage des courbes de perte
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.legend()
plt.title("Courbes d'apprentissage")
plt.savefig("courbes_apprentissage.png",format="png")


# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(val_targets, val_preds)

# Affichage avec seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.savefig("confusion_matrix.png",format="png")
