import pandas as pd
import matplotlib.pyplot as plt
import torch

# Chargement des données
data_path = "./data"
df_train = pd.read_csv(f'{data_path}/X_train/X_train')
df_test = pd.read_csv(f'{data_path}/X_test/X_test')
y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None, names=['is_letter'])

# Utiliser l'ensemble des sujets (15 au total)
all_subjects = df_train['sujeto'].unique()  # récupère dynamiquement tous les sujets présents
df_train = df_train[df_train['sujeto'].isin(all_subjects)]
df_test = df_test[df_test['sujeto'].isin(all_subjects)]
y_train = y_train.iloc[df_train.index]


# Identification des canaux à partir des noms de colonnes courts (< 5 caractères)
channels = list(set([c[:2] for c in df_train.columns if len(c) < 5]))
channels.sort()

# 1. Nombre total de lignes dans le train set
n_rows = df_train.shape[0]
print("Nombre de lignes dans le train set :", n_rows)

# 2. Nombre de lignes par sujet
subject_counts = df_train['sujeto'].value_counts().sort_index()
print("Nombre de lignes par sujet :")
print(subject_counts)

# 3. Longueur des enregistrements par canal
# Ici, on considère que la "longueur" correspond au nombre de colonnes pour lesquelles le nom commence par le préfixe du canal.
channel_lengths = {ch: sum([1 for c in df_train.columns if c.startswith(ch)]) for ch in channels}
print("Longueur des enregistrements par canal :")
print(channel_lengths)

# 4. Visualisation des statistiques dans une figure avec 2 subplots
fig, ax = plt.subplots(figsize=(8, 5))  # Un seul subplot

# Graphique pour le nombre de lignes par sujet
subject_counts.plot(kind='bar', ax=ax)
ax.set_title("Nombre de lignes par sujet")
ax.set_xlabel("Sujet")
ax.set_ylabel("Nombre de lignes")

plt.tight_layout()
plt.savefig("stats_dataset.png")
plt.close()

print("La figure des statistiques a été enregistrée sous 'stats_dataset.png'")
