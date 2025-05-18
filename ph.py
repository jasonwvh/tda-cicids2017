import kmapper as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from gtda.diagrams import PersistenceLandscape
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence, EuclideanCechPersistence, SparseRipsPersistence
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding, SlidingWindow
from kmapper import Cover
from numpy.ma.core import equal
from scipy.stats import mode
from sklearn import ensemble, cluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors


def load_and_prep_data():
    columns_to_keep = [
        ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp',
        ' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
        ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
        'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
        ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
    ]
    df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                     encoding='ISO-8859-1', usecols=columns_to_keep, low_memory=False)
    df.columns = df.columns.str.strip()

    highest_non_inf = df.loc[df['Flow Bytes/s'] != np.inf, 'Flow Bytes/s'].max()
    df.replace(np.inf, highest_non_inf, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.dropna()

    df["Source IP"] = df["Source IP"].apply(lambda x: float(str(x).replace(".", "")))
    df["Destination IP"] = df["Destination IP"].apply(lambda x: float(str(x).replace(".", "")))
    df['Flow Duration'] = df['Flow Duration'].clip(lower=1e-6)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    label_map = {"BENIGN": 0, "Web Attack  Brute Force": 11, "Web Attack  XSS": 12, "Web Attack  Sql Injection": 13}
    df["Label"] = df["Label"].map(label_map)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

df = load_and_prep_data()

## Aggregation
def safe_mean(x):
    return np.mean(x.replace([np.inf, -np.inf], np.nan)) if not x.replace([np.inf, -np.inf], np.nan).isna().all() else np.nan
def safe_sum(x):
    return np.sum(x.replace([np.inf, -np.inf], np.nan)) if not x.replace([np.inf, -np.inf], np.nan).isna().all() else np.nan

resampled_data = df.resample('60s').agg(
    flow_count=('Source IP', 'size'),
    flow_duration=('Flow Duration', safe_mean),
    flow_bytes=('Flow Bytes/s', safe_sum),
    flow_packets=('Flow Packets/s', safe_sum),
    flow_mean=('Flow IAT Mean', safe_mean),
    fwd_mean=('Fwd IAT Mean', safe_mean),
    bwd_mean=('Bwd IAT Mean', safe_mean),
    active_mean=('Active Mean', safe_mean),
    idle_mean=('Idle Mean', safe_mean),
    label=('Label', 'max'),
)

## no separate
features = ['flow_duration', 'flow_mean', 'fwd_mean', 'bwd_mean', 'active_mean', 'idle_mean']
featured_df = resampled_data[features]
dim = featured_df.columns.size

scaler = RobustScaler()
df_scaled = scaler.fit_transform(featured_df)
#
# umap_reducer = umap.UMAP(n_components=3, random_state=42)
# pca = PCA(n_components=3)
# df_reduced = umap_reducer.fit_transform(df_scaled)

te = TakensEmbedding(time_delay=1, dimension=dim)
embedding = te.fit_transform(featured_df)
embedding = embedding.reshape(featured_df.index.size, dim)

ph = VietorisRipsPersistence(homology_dimensions=[0, 1])
embedding_reshaped = embedding[None, :, :]
diagrams = ph.fit_transform(embedding_reshaped)[0]

colors = {0: 'green', 1: 'blue', 2: 'purple'}
markers = {0: 'x', 1: 'o', 2: '^'}
labels = {0: 'H0', 1: 'H1', 2: 'H2'}
max_death = 0

fig = plt.subplots(figsize=(6, 6))
for dim in [1]:
    dgm = diagrams[diagrams[:, 2] == dim]
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], c=colors[dim],
                    label=f'{labels[dim]}', alpha=1)
        max_death = max(max_death, max(dgm[:, 1]))
plt.plot([0, max_death], [0, max_death], 'k--', alpha=0.5)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('Persistence Diagrams (H0, H1, H2) no separate umap/pca')
plt.legend()
plt.grid(True)
plt.show()


## Separate Benign and Attack
features = ['flow_duration', 'flow_mean', 'fwd_mean', 'bwd_mean', 'active_mean', 'idle_mean']
benigns = resampled_data[resampled_data['label'] == 0][features]
attacks = resampled_data[resampled_data['label'] != 0][features]

## ph on aggregated
scaler = StandardScaler()
benigns_scaled = scaler.fit_transform(benigns[features])
attacks_scaled = scaler.fit_transform(attacks[features])

umap_reducer = umap.UMAP(n_components=3, random_state=42)
pca = PCA(n_components=3)
benigns_reduced = umap_reducer.fit_transform(benigns_scaled)
attacks_reduced = umap_reducer.fit_transform(attacks_scaled)

benigns_reshaped = benigns_reduced[None, :, :]
attacks_reshaped = attacks_reduced[None, :, :]

diagrams_benigns_ori = ph.fit_transform(benigns_reshaped)
diagrams_attacks_ori = ph.fit_transform(attacks_reshaped)
diagrams_benigns = diagrams_benigns_ori[0]
diagrams_attacks = diagrams_attacks_ori[0]

colors_benigns = {0: 'blue', 1: 'green', 2: 'darkgreen'}
colors_attacks = {0: 'purple', 1: 'red', 2: 'darkred'}
markers = {0: 'x', 1: 'o', 2: '^'}
labels = {0: 'H0', 1: 'H1', 2: 'H2'}
max_death = 0
for dim in [1,2]:
    dgm = diagrams_benigns[diagrams_benigns[:, 2] == dim]
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], c=colors_benigns[dim], marker=markers[dim],
                    label=f'{labels[dim]} Benign', alpha=0.5)
        max_death = max(max_death, max(dgm[:, 1]))
for dim in [1,2]:
    dgm = diagrams_attacks[diagrams_attacks[:, 2] == dim]
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], c=colors_attacks[dim], marker=markers[dim],
                    label=f'{labels[dim]} Attack', alpha=1)
        max_death = max(max_death, max(dgm[:, 1]))

plt.plot([0, max_death], [0, max_death], 'k--', alpha=0.5)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('Persistence Diagrams (H0, H1, H2) pca/umap')
plt.legend()
plt.grid(True)
plt.show()

# Multi dimensional Takens with separate benigns and attacks
dim=benigns.columns.size
te = TakensEmbedding(time_delay=1, dimension=dim)
embedding_benigns = te.fit_transform(benigns)
embedding_benigns = embedding_benigns.reshape(benigns.index.size, dim)
embedding_attacks = te.fit_transform(attacks)
embedding_attacks = embedding_attacks.reshape(attacks.index.size, dim)

benigns_reshaped = embedding_benigns[None, :, :]
attacks_reshaped = embedding_attacks[None, :, :]
diagrams_benigns_ori = ph.fit_transform(benigns_reshaped)
diagrams_attacks_ori = ph.fit_transform(attacks_reshaped)

diagrams_benigns = diagrams_benigns_ori[0]
diagrams_attacks = diagrams_attacks_ori[0]

colors_benigns = {0: 'blue', 1: 'green', 2: 'darkgreen'}
colors_attacks = {0: 'purple', 1: 'red', 2: 'darkred'}
markers = {0: 'x', 1: 'o', 2: '^'}
labels = {0: 'H0', 1: 'H1', 2: 'H2'}
max_death = 0

fig = plt.subplots(figsize=(6, 6))
for dim in [1]:
    dgm = diagrams_benigns[diagrams_benigns[:, 2] == dim]
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], c=colors_benigns[dim], marker=markers[dim],
                    label=f'{labels[dim]} Benign', alpha=0.5)
        max_death = max(max_death, max(dgm[:, 1]))
for dim in [1]:
    dgm = diagrams_attacks[diagrams_attacks[:, 2] == dim]
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], c=colors_attacks[dim], marker=markers[dim],
                    label=f'{labels[dim]} Attack', alpha=1)
        max_death = max(max_death, max(dgm[:, 1]))
plt.plot([0, max_death], [0, max_death], 'k--', alpha=0.5)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('Persistence Diagrams (H0, H1, H2) takens')
plt.legend()
plt.grid(True)
plt.show()

pl = PersistenceLandscape()
landscapes_benigns = pl.fit_transform(diagrams_benigns_ori)
fig = plt.figure(figsize=(10, 8))
for i in range(landscapes_benigns.shape[1]):
    plt.plot(landscapes_benigns[0, i], label=f'Landscape {i}')
plt.xlabel('Filtration value')
plt.ylabel('Landscape value')
plt.legend()
plt.title('Persistence Landscapes')
plt.show()

landscapes_attacks = pl.fit_transform(diagrams_attacks_ori)
fig = plt.figure(figsize=(10, 8))
for i in range(landscapes_attacks.shape[1]):
    plt.plot(landscapes_attacks[0, i], label=f'Landscape {i}')
plt.xlabel('Filtration value')
plt.ylabel('Landscape value')
plt.legend()
plt.title('Persistence Landscapes')
plt.show()
