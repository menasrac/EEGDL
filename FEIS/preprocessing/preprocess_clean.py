import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt,lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import torch
from kymatio.torch import Scattering1D
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Paramètres généraux
dossier = "../experiments/"
sujets = ["01","02","03"]  # Par exemple
fs = 256
lowcut, highcut = 10 , 100
taille_standard = 1280
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
    grouped = df.groupby("Epoch")
    #print(grouped.head())
    #print(grouped.size)
    
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

# 3) Construction du dataset global
all_samples = []
for sujet in sujets:
    blocks = read_csv_for_subject(sujet)
    all_samples.extend(blocks)


X_list, y_list = zip(*all_samples)
X_list = np.array(X_list, dtype=object)
y_list = np.array(y_list)

# 4) Split train/test + normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X_list, y_list, test_size=0.2, stratify=y_list, random_state=42
)

train_concat = np.concatenate(X_train, axis=0)
scalers = [StandardScaler().fit(train_concat[:, c].reshape(-1, 1)) for c in range(nb_canaux)]

def normalize_block(block):
    # Normalisation canal par canal
    return np.column_stack([
        scalers[c].transform(block[:, c].reshape(-1, 1)).ravel()
        for c in range(nb_canaux)
    ])

X_train_norm = [normalize_block(b) for b in X_train]
X_test_norm  = [normalize_block(b) for b in X_test]

# 5) Wavelet Scattering Transform (GPU) en conservant la dimension temporelle
def wavelet_scattering_transform_gpu(block, J=6, Q=8):
    """
    block shape: (1280, nb_canaux)
    Retourne un tenseur numpy de forme (nb_canaux, n_coeffs, T_scattered)
    """
    T = block.shape[0]
    block_np = np.array(block.T, dtype=np.float32)  # (nb_canaux, 1280)
    block_torch = torch.tensor(block_np, dtype=torch.float32, device='cuda')
    
    sc = Scattering1D(J=J, shape=(T,), Q=Q).cuda()
    
    # Pour chaque canal: sc(...) -> shape: (1, n_coeffs, T_scattered)
    # On retire la dimension batch => (n_coeffs, T_scattered)
    feats_all = []
    for c in range(block_torch.shape[0]):
        Sx = sc(block_torch[c].view(1, -1).contiguous())  # shape (1, n_coeffs, T')
        Sx = Sx.squeeze(0)  # shape (n_coeffs, T')
        feats_all.append(Sx.cpu().numpy())
    
    # feats_all est une liste de (n_coeffs, T_scattered) pour chaque canal
    # On empile => shape (nb_canaux, n_coeffs, T_scattered)
    feats_all = np.stack(feats_all, axis=0)
    
    return feats_all  # Pas de mean(), ni de normalisation finale

X_train_feats = Parallel(n_jobs=-1)(
    delayed(wavelet_scattering_transform_gpu)(b)
    for b in X_train_norm if b is not None
)
X_test_feats = Parallel(n_jobs=-1)(
    delayed(wavelet_scattering_transform_gpu)(b)
    for b in X_test_norm if b is not None
)

X_train_feats = np.array([x for x in X_train_feats if x is not None], dtype=object)
X_test_feats  = np.array([x for x in X_test_feats if x is not None], dtype=object)

# Ici, chaque bloc a shape (nb_canaux, n_coeffs, T_scattered).
df_train = pd.DataFrame({"Features": list(X_train_feats), "Label": y_train})
df_test  = pd.DataFrame({"Features": list(X_test_feats),  "Label": y_test})
df_train.to_pickle("train_scattering.pkl")
df_test.to_pickle("test_scattering.pkl")
print("Pipeline terminé.")

# DataLoader PyTorch
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # X[idx] shape => (nb_canaux, n_coeffs, T_scattered)
        # Selon ton modèle, tu transformeras en torch.float32
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

train_dataset = EEGDataset(X_train_feats, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print("Train loader prêt.")

