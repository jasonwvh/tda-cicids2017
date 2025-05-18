import os
import kmapper as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gtda.diagrams import PersistenceLandscape
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence, EuclideanCechPersistence, SparseRipsPersistence
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding, SlidingWindow
from kmapper import Cover
from numpy.ma.core import equal
from scipy.stats import mode
from sklearn import ensemble, cluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Ensure output directory exists
os.makedirs("diagrams", exist_ok=True)

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

aggregation_interval = '60s'
resampled_data = df.resample(aggregation_interval).agg(
    flow_count=('Source IP', 'size'),
    flow_duration=('Flow Duration', safe_sum),
    flow_bytes=('Flow Bytes/s', safe_sum),
    flow_packets=('Flow Packets/s', safe_sum),
    flow_mean=('Flow IAT Mean', safe_mean),
    fwd_mean=('Fwd IAT Mean', safe_mean),
    bwd_mean=('Bwd IAT Mean', safe_mean),
    active_mean=('Active Mean', safe_mean),
    idle_mean=('Idle Mean', safe_mean),
    label=('Label', 'max'),
)

features = ['flow_duration', 'flow_mean', 'fwd_mean', 'bwd_mean', 'active_mean', 'idle_mean']
featured_df = resampled_data[features]
dim = len(features)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(featured_df)
df_scaled = pd.DataFrame(df_scaled, index=featured_df.index, columns=features)

window_size = pd.Timedelta(seconds=900)
step_size = pd.Timedelta(seconds=60)
window_samples = int(window_size / pd.Timedelta(aggregation_interval))
step_samples = int(step_size / pd.Timedelta(aggregation_interval))

# te = SingleTakensEmbedding('fixed', time_delay=1, dimension=3)
te = TakensEmbedding(time_delay=1, dimension=dim)
ph = VietorisRipsPersistence(homology_dimensions=[0, 1])
embedding = te.fit_transform(featured_df)

colors = {0: 'green', 1: 'blue', 2: 'red'}
markers = {0: 'x', 1: 'o', 2: 's'}
labels = {0: 'H0', 1: 'H1', 2: 'H2'}

landscape_features = []
landscape_l2norm = []
for window_idx in range(0, len(df_scaled) - window_samples + 1, step_samples):
    window_data = df_scaled.iloc[window_idx:window_idx + window_samples]
    if len(window_data) < window_samples:
        continue

    if window_data.index[0] not in [pd.Timestamp('2017-06-07 09:01:00'), pd.Timestamp('2017-06-07 09:15:00'), pd.Timestamp('2017-06-07 09:30:00'), pd.Timestamp('2017-06-07 09:45:00'), pd.Timestamp('2017-06-07 10:00:00'), pd.Timestamp('2017-06-07 10:15:00'), pd.Timestamp('2017-06-07 10:40:00'), pd.Timestamp('2017-06-07 11:00:00'), pd.Timestamp('2017-06-07 11:30:00'),]:
        continue

    try:
        # single takens
        # embedding = te.fit_transform(window_data['flow_bytes'])

        embedding = te.fit_transform(window_data.values)
        embedding = embedding.reshape(window_data.index.size, 6)
    except ValueError as e:
        print(f"Skipping window {window_idx} due to embedding error: {e}")
        continue

    pca = PCA(n_components=3)
    embedding = pca.fit_transform(embedding)

    embedding_reshaped = embedding[None, :, :]
    try:
        # embedding_reshaped = window_data.values[None, :, :]
        diagrams_ori = ph.fit_transform(embedding_reshaped)
        diagrams = diagrams_ori[0]
    except ValueError as e:
        print(f"Skipping window {window_idx} due to persistence error: {e}")
        continue

    fig, ax = plt.subplots(figsize=(6, 6))
    max_death = 0
    for dim in [0, 1]:
        dgm = diagrams[diagrams[:, 2] == dim]
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], c=colors[dim], label=f'{labels[dim]}', alpha=1)
            max_death = max(max_death, max(dgm[:, 1]))

    ax.plot([0, max_death], [0, max_death], 'k--', alpha=0.5)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'Persistence Diagram - Window {window_idx} ({window_data.index[0]})')
    ax.legend()
    ax.grid(True)

    plt.savefig(f"diagrams/diagram_window_{window_idx}.png")

    pl = PersistenceLandscape()
    landscapes = pl.fit_transform(diagrams_ori)
    fig = plt.figure(figsize=(10, 8))
    for i in range(landscapes.shape[1]):
        plt.plot(landscapes[0, i], label=f'Landscape {i}')
    plt.xlabel('Filtration value')
    plt.ylabel('Landscape value')
    plt.legend()
    plt.title(f'Persistence Landscapes - Window {window_idx} ({window_data.index[0]})')
    plt.savefig(f"diagrams/landscape_window_{window_idx}.png")
    plt.close(fig)

    ## print the actual persistence number
    flattened = landscapes.flatten()
    landscape_features.append(flattened)
    summary_value = np.sum(np.abs(flattened))
    landscape_l2norm.append(summary_value)

X = np.array(landscape_features)
Y = np.array(landscape_l2norm)
print(Y)
print("Persistence diagrams saved in 'diagrams/' directory.")