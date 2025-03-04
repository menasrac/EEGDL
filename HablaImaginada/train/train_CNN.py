import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données à partir du pickle
with open("old/scattering_features_7sujets.pkl", "rb") as f:
    data_dict = pickle.load(f)

X_train = data_dict["X_train"]  # forme : (n_samples, n_channels, n_time_steps)
X_test = data_dict["X_test"]
y_train = data_dict["y_train"].astype(np.float32)

print("Forme des données d'entraînement :", X_train.shape)

# Vérifier la distribution des classes
unique, counts = np.unique(y_train, return_counts=True)
print("Distribution des classes:", dict(zip(unique, counts)))

n_samples, n_channels, n_time_steps = X_train.shape


# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Création d'un dataset et découpage en train/validation
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Architecture CNN simplifiée pour données normalisées
class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels, output_dim=1):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Après deux couches de max pooling, la dimension temporelle est divisée par 4
        final_time_dim = n_time_steps // (2**2)
        self.fc1 = nn.Linear(64 * final_time_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.2)  # dropout à 20%

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Aplatir
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Retourne les logits
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN1DClassifier(input_channels=n_channels).to(device)

# Utiliser BCEWithLogitsLoss (la Sigmoid sera appliquée dans l'évaluation)
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
            probs = torch.sigmoid(outputs)
            preds = probs.round()
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, all_preds, all_targets

num_epochs = 20  # Entraînement sur plus d'époques pour voir une évolution
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    grad_norm_sum = 0.0
    batch_count = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        # Calcul de la norme des gradients pour vérifier leur circulation
        batch_grad_norm = sum(param.grad.data.norm(2).item() for param in model.parameters() if param.grad is not None)
        grad_norm_sum += batch_grad_norm
        batch_count += 1
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    avg_grad_norm = grad_norm_sum / batch_count
    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_preds, val_targets = evaluate(model, val_loader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}")
    print(classification_report(val_targets, val_preds, digits=4))

torch.save(model.state_dict(), "cnn_model_normalized.pth")
print("Modèle sauvegardé dans cnn_model_normalized.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.legend()
plt.title("Courbes d'apprentissage")
plt.savefig("learning_curves_cnn_normalized.png", format="png")

conf_matrix = confusion_matrix(val_targets, val_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.title("Matrice de confusion")
plt.savefig("confusion_matrix_cnn_normalized.png", format="png")
