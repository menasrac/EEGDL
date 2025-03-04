import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from kymatio.torch import Scattering1D
from scipy.signal import butter, filtfilt
import pickle


# Version standard du preprocessing avec la scattering transform moyennée dans le temps

# Charger les données
data_path = "./data"
df_train = pd.read_csv(f'{data_path}/X_train/X_train')
df_test = pd.read_csv(f'{data_path}/X_test/X_test')
y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None, names=['is_letter'])
# Sélectionner un ou plusieurs sujets
selected_subjects = ['C','A','B','D','E']  # Remplacez par la liste des sujets que vous souhaitez analyser, par ex. ['A', 'B']

# Filtrer les données pour ne garder que les sujets sélectionnés
df_train = df_train[df_train['sujeto'].isin(selected_subjects)]
df_test = df_test[df_test['sujeto'].isin(selected_subjects)]
print(df_train.head())
print(df_train.columns)
y_train = y_train.iloc[df_train.index]  # Aligner les indices de y_train avec df_train

# Dispositif (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


channels = list(set([c[:2] for c in df_train.columns if len(c) < 5]))
channels.sort()
#plot_eeg_samples(df_train,y_train,channels,"visu_chan.png")

# Normalisation par canal
def normalize_channels(data):
    """Normalisation canal par canal"""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Filtrage passe-bande (10-100Hz)
def bandpass_filter(data, lowcut=2, highcut=40, fs=1024.0, order=4):
    """Applique un filtre passe-bande sur les données EEG"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Wavelet Scattering Transform pour MLP
def apply_wavelet_scattering(data, J=6, Q=8):
    """Applique la transformée de scattering sur les données EEG"""
    # S'assurer que le tableau est contigu (pour éviter les strides négatifs)
    data = np.ascontiguousarray(data)
    scattering = Scattering1D(J=J, Q=Q, shape=(data.shape[-1],))
    data_scattered = scattering(torch.tensor(data, dtype=torch.float32)).cpu().numpy()
    return data_scattered


# Prétraitement des données
df_train = df_train.drop(columns=["sujeto"])
X_train_norm = normalize_channels(df_train.values)
df_test = df_test.drop(columns=["sujeto"])
X_test_norm = normalize_channels(df_test.values)

df_train_norm = pd.DataFrame(X_train_norm, columns=df_train.columns)
df_test_norm = pd.DataFrame(X_test_norm, columns=df_test.columns)

#plot_eeg_samples(df_train_norm, y_train,channels,"visu_chan_normalized.png")

# Application du filtre passe-bande (avant les wavelets)
X_train_filtered = bandpass_filter(X_train_norm)
X_test_filtered = bandpass_filter(X_test_norm)
df_train_filtered = pd.DataFrame(X_train_filtered,columns=df_train.columns)
#plot_eeg_samples(df_train_filtered, y_train,channels,"visu_chan_filtered.png")


# Appliquer la transformation wavelet après le filtrage
X_train_scattered = apply_wavelet_scattering(X_train_filtered)
X_test_scattered = apply_wavelet_scattering(X_test_filtered)

# Afficher la forme du scattering (nombre de coefficients)
print("Forme du scattering (train):", X_train_scattered.shape)
if X_train_scattered.ndim == 2:
    print("Nombre de coefficients par échantillon :", X_train_scattered.shape[1])
else:
    print("Scattering output shape:", X_train_scattered.shape)


def plot_scattering_features(df_filtered, channels, path, J=6, Q=8):
    """
    Pour chaque canal, on extrait les colonnes correspondant à ce canal dans le DataFrame filtré,
    puis on prend le premier échantillon et on applique la transformation de scattering sur ce canal.
    Ensuite, on trace sur un sous-graphique les 8 premiers coefficients.
    """
    n_channels = len(channels)
    fig, axes = plt.subplots(nrows=n_channels, ncols=1, figsize=(8, 2 * n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i, ch in enumerate(channels):
        # Sélectionner les colonnes du canal via une regex (colonnes commençant par le préfixe)
        ch_data = df_filtered.filter(regex=f"^{ch}")
        # Prendre le premier échantillon
        sample = ch_data.iloc[0].values
        sample = np.ascontiguousarray(sample)  # S'assurer que le tableau est contigu
        # Créer l'objet scattering pour ce signal
        scattering = Scattering1D(J=J, Q=Q, shape=sample.shape)
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        sample_scattered = scattering(sample_tensor).cpu().numpy().flatten()
        # Afficher le nombre de coefficients pour ce canal
        print(f"Canal {ch} : {sample_scattered.shape[0]} coefficients de scattering")
        # Tracer les 8 premières features
        axes[i].plot(range(8), sample_scattered[:8], marker='o')
        axes[i].set_title(f"Scattering features - Canal {ch}")
        axes[i].set_xlabel("Indice de coefficient")
        axes[i].set_ylabel("Valeur")
    
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

plot_scattering_features(df_train_filtered, channels, "scattering_features_aggregated.png", J=6, Q=8)


# Sauvegarder les données et labels dans un fichier pickle
data_dict = {
    "X_train": X_train_scattered,  # np.ndarray shape (n_samples, n_coeffs)
    "X_test": X_test_scattered,
    "y_train": y_train.values  # Par exemple, un array de shape (n_samples, 1)
}

with open("scattering_features_5sujets.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("Sauvegarde effectuée dans scattering_features_5sujets.pkl")

