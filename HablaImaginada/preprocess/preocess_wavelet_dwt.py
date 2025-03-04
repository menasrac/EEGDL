import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.signal import butter, filtfilt, resample
import pickle
import concurrent.futures

# Charger les données
data_path = "./data"
df_train = pd.read_csv(f'{data_path}/X_train/X_train')
df_test = pd.read_csv(f'{data_path}/X_test/X_test')
y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None, names=['is_letter'])

selected_subjects = ['C','F','G','E','I','A','B','H','D']# On ajuste le nombre et le choix des sujets
df_train = df_train[df_train['sujeto'].isin(selected_subjects)]
df_test = df_test[df_test['sujeto'].isin(selected_subjects)]
y_train = y_train.iloc[df_train.index]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channels = list(set([c[:2] for c in df_train.columns if len(c) < 5]))
channels.sort()

def normalize_channels(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def bandpass_filter(data, lowcut=2, highcut=40, fs=1024.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def resample_signal(data, original_fs=1024, target_fs=128):
    factor = target_fs / original_fs
    new_length = int(len(data) * factor)
    return resample(data, new_length)

def process_signal(signal, wavelet='db4', level=5):
    return np.concatenate(pywt.wavedec(signal, wavelet, level=level))

def apply_dwt_decomposition_parallel(data, wavelet='db4', level=5):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        transformed_signals = list(executor.map(process_signal, data))
    return np.array(transformed_signals)


df_train = df_train.drop(columns=["sujeto"])
X_train_norm = normalize_channels(df_train.values)
df_test = df_test.drop(columns=["sujeto"])
X_test_norm = normalize_channels(df_test.values)

X_train_filtered = bandpass_filter(X_train_norm)
X_test_filtered = bandpass_filter(X_test_norm)

X_train_resampled = np.array([resample_signal(sig) for sig in X_train_filtered])
X_test_resampled = np.array([resample_signal(sig) for sig in X_test_filtered])

X_train_dwt = apply_dwt_decomposition_parallel(X_train_resampled)
X_test_dwt = apply_dwt_decomposition_parallel(X_test_resampled)

print("Forme de la décomposition DWT (train):", X_train_dwt.shape)

# Visualisation de la décomposition DWT sur un signal
sample_signal = X_train_resampled[0]
coeffs = pywt.wavedec(sample_signal, 'db4', level=5)
num_levels = len(coeffs)

plt.figure(figsize=(10, num_levels * 2))
plt.subplot(num_levels + 1, 1, 1)
plt.plot(sample_signal)
plt.title("Signal original")

for i, coeff in enumerate(coeffs):
    plt.subplot(num_levels + 1, 1, i + 2)
    plt.plot(coeff)
    plt.title(f"Composant DWT - Niveau {i}")

plt.tight_layout()
plt.savefig("decompo_dwt.png", format="png")

data_dict = {
    "X_train": X_train_dwt,
    "X_test": X_test_dwt,
    "y_train": y_train.values
}

with open("dwt_features_9sujets.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("Sauvegarde effectuée dans dwt_features_9sujets.pkl")
