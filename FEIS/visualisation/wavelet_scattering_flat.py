import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import torch
from kymatio.torch import Scattering1D
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Paramètres généraux pour 256 Hz
dossier = "../experiments/"
sujets = ["01"]  # Par exemple
fs = 256
lowcut, highcut = 10, 100
taille_standard = 1280   # 5 secondes à 256 Hz
nb_canaux = 14

# Filtrage passe-bande
def bandpass_filter(data, lowcut=10, highcut=100, fs=256, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Lecture et découpage
def read_csv_for_subject(sujet):
    chemin_csv = os.path.join(dossier, sujet, "thinking.csv")
    if not os.path.exists(chemin_csv):
        print(f"Fichier non trouvé: {chemin_csv}")
        return []
    
    df = pd.read_csv(chemin_csv)
    df["New_Block"] = df["Label"].ne(df["Label"].shift()).cumsum()
    grouped = df.groupby("New_Block")
    
    samples = []
    for block_id, block in grouped:
        label = block["Label"].iloc[0]
        eeg_data = block.iloc[:, 2:2+nb_canaux].values
        length = eeg_data.shape[0]
        if length < taille_standard:
            continue
        elif length > 2 * taille_standard:
            continue
        elif length == taille_standard:
            eeg_data = bandpass_filter(eeg_data, lowcut, highcut, fs)
            samples.append((eeg_data, label))
        else:
            num_sub = length // taille_standard
            for i in range(num_sub):
                sub_block = eeg_data[i*taille_standard:(i+1)*taille_standard]
                sub_block = bandpass_filter(sub_block, lowcut, highcut, fs)
                samples.append((sub_block, label))

    return samples

# Construction du dataset global
all_samples = []
for sujet in sujets:
    blocks = read_csv_for_subject(sujet)
    all_samples.extend(blocks)

X_list, y_list = zip(*all_samples)
X_list = np.array(X_list, dtype=object)
y_list = np.array(y_list)

# Split train/test + normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X_list, y_list, test_size=0.2, stratify=y_list, random_state=42
)

# Concaténation de l'entraînement pour calculer le StandardScaler canal par canal
train_concat = np.concatenate(X_train, axis=0)  # shape (N * taille_standard, nb_canaux)
scalers = [
    StandardScaler().fit(train_concat[:, c].reshape(-1, 1))
    for c in range(nb_canaux)
]

def normalize_block(block):
    # Normalisation canal par canal
    return np.column_stack([
        scalers[c].transform(block[:, c].reshape(-1, 1)).ravel()
        for c in range(nb_canaux)
    ])

X_train_norm = [normalize_block(b) for b in X_train]
X_test_norm  = [normalize_block(b) for b in X_test]

# Wavelet Scattering "flat" : concaténation de tous les canaux en 1D
def wavelet_scattering_transform_flat(block, J=6, Q=8):
    """
    block shape: (taille_standard, nb_canaux)
    On aplatit => (taille_standard * nb_canaux,)
    Retourne un tenseur numpy de forme (n_coeffs, T_scattered).
    """
    # Flatten en 1D
    flat = block.flatten()  # shape (taille_standard*nb_canaux,)
    flat_torch = torch.tensor(flat, device='cuda', dtype=torch.float32).unsqueeze(0)
    # shape (batch=1, taille_standard*nb_canaux)

    sc = Scattering1D(J=J, shape=(flat.shape[0],), Q=Q).cuda()
    Sx = sc(flat_torch)  # shape (1, n_coeffs, T_scattered)

    Sx = Sx.squeeze(0).cpu().numpy()  # shape (n_coeffs, T_scattered)
    return Sx

# Application parallèle de la scattering transform
X_train_feats = Parallel(n_jobs=-1)(
    delayed(wavelet_scattering_transform_flat)(b)
    for b in X_train_norm
)
X_test_feats = Parallel(n_jobs=-1)(
    delayed(wavelet_scattering_transform_flat)(b)
    for b in X_test_norm
)

X_train_feats = np.array(X_train_feats, dtype=object)
X_test_feats  = np.array(X_test_feats,  dtype=object)

# Sauvegarde dans DataFrame
df_train = pd.DataFrame({"Features": list(X_train_feats), "Label": y_train})
df_test  = pd.DataFrame({"Features": list(X_test_feats),  "Label": y_test})

df_train.to_pickle("train_scattering.pkl")
df_test.to_pickle("test_scattering.pkl")
print("Pipeline terminé.")

# Visualisation d'un exemple
example_idx = 0
feats = X_train_feats[example_idx]  # shape (n_coeffs, T_scattered)
print("feats", feats)

plt.figure(figsize=(10,6))
for i in range(8):
    plt.plot(feats[i, :], label=f"Feature {i+1}")
plt.title("Exemple de coefficients (Wavelet Scattering) sur un bloc")
plt.xlabel("Index temporel (après scattering)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("flatscat_256.png", format='png')



