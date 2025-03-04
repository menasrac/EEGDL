import pandas as pd
from scipy import signal
import os
import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 18
})

dossier = "../experiments/07"
chemin_csv = dossier + "/thinking.csv"

# Charger le fichier
df = pd.read_csv(chemin_csv)

# Identifier les blocs (changement de phonème)
df["New_Block"] = df["Epoch"].ne(df["Epoch"].shift()).cumsum()

# Sélectionner un bloc du label "f"
df_f = df[df["Label"] == "f"]
# Create signals directory if it doesn't exist
os.makedirs(f"signals", exist_ok=True)

# Get all unique blocks for "f"
blocks = df_f["New_Block"].unique()

# Process each block separately
for block_num in blocks:
    # Get data for current block
    block_data = df_f[df_f["New_Block"] == block_num]
    
    # Extract time and EEG channels
    time = block_data["Time:256Hz"]
    eeg_channels = block_data.iloc[:, 2:3]  # choix canaux

    # Normalisation
    normalised = (eeg_channels - eeg_channels.mean(axis=0)) / (eeg_channels.std(axis=0) + 1e-8)

    # Design bandpass filter
    fs = 256  # Sampling frequency (Hz)
    nyquist = fs/2
    low = 10/nyquist
    high = 100/nyquist
    b, a = signal.butter(4, [low, high], btype='bandpass')

    # Apply filter to each channel
    banded = pd.DataFrame(
        signal.filtfilt(b, a, normalised, axis=0),
        columns=normalised.columns,
        index=normalised.index
    )

    # Create a single plot with all channels and processing stages
    fig = plt.figure(figsize=(15, 10))
    
    # Plot all channels for each processing stage
    ax1 = plt.subplot(3, 1, 1)
    for channel in eeg_channels.columns:
        ax1.plot(time, eeg_channels[channel], label=channel)
    ax1.set_title(f"Raw EEG Signals - Block {block_num}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (uV)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = plt.subplot(3, 1, 2)
    for channel in normalised.columns:
        ax2.plot(time, normalised[channel], label=channel)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Normalized Signals")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    ax3 = plt.subplot(3, 1, 3)
    for channel in banded.columns:
        ax3.plot(time, banded[channel], label=channel)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Filtered Signals")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    # Get handles and labels from any of the subplots (they're the same)
    handles, labels = ax1.get_legend_handles_labels()
    
    # Create a single legend outside the plots
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig(f"signals/block_{block_num}_all_stages.png", bbox_inches='tight')
    plt.close()


