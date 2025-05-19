from threading import active_count

import kmapper as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
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

    label_map = {"BENIGN": 0, "Web Attack ï¿½ Brute Force": 11, "Web Attack ï¿½ XSS": 12, "Web Attack ï¿½ Sql Injection": 13}
    df["Label"] = df["Label"].map(label_map)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

df = load_and_prep_data()


## Plotting Raw Data
# benigns = df[df['Label'] == 0]
# attacks = df[df['Label'] != 0]
# brutes = df[df['Label'] == 11]
# xss = df[df['Label'] == 12]
# sqli = df[df['Label'] == 13]
#
# features = ['Flow Duration', 'Flow IAT Mean', 'Active Mean']
# fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
# for i, col in enumerate(features):
#     axs[i].plot(benigns.index, benigns[col], color='green', linewidth=1, alpha=0.5, label='Benign')
#     axs[i].plot(brutes.index, brutes[col], color='red', linewidth=2, label='Brute Force')
#     axs[i].plot(xss.index, xss[col], color='purple', linewidth=2, label='XSS')
#     axs[i].plot(sqli.index, sqli[col], color='orange', linewidth=2, label='SQL Injection')
#     axs[i].set_title(f'{col}: Benign vs Attack')
#     axs[i].set_ylabel(col)
#     axs[i].legend()
# plt.xlabel('Flow (microseconds)')
# plt.tight_layout()
# plt.show()

## Mapper on Raw Input
# features = [c for c in df.columns if c not in
#             ['Source IP',
#              'Source Port',
#              'Destination IP',
#              'Destination Port',
#              'Protocol',
#              'Timestamp',
#              'Label']]
#
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.dropna(inplace=True)
# X = df[features].values
# y = df['Label'].values
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# mapper = km.KeplerMapper(verbose=1)
# lens1 = mapper.fit_transform(X_scaled, projection="l2norm")
#
# projector = ensemble.IsolationForest(random_state=42)
# projector.fit(X)
# lens2 = projector.decision_function(X_scaled)
#
# lens = np.c_[lens1, lens2]

# fig, axs = plt.subplots(1, 2, figsize=(9,4))
# axs[0].scatter(lens1,lens2,c=y.reshape(-1,1),alpha=0.3)
# axs[0].set_xlabel('L^2-Norm')
# axs[0].set_ylabel('PCA')
# axs[1].scatter(lens1,lens3,c=y.reshape(-1,1),alpha=0.3)
# axs[1].set_xlabel('L^2-Norm')
# axs[1].set_ylabel('IsolationForest')
# plt.tight_layout()
# plt.show()
#
# cover = Cover(n_cubes=20, perc_overlap=0.20)
# clusterer = DBSCAN(eps=1, min_samples=2)
# G = mapper.map(
#     lens,
#     X_scaled,
#     cover=cover,
#     clusterer=clusterer,
# )
#
# _ = mapper.visualize(
#     G,
#     custom_tooltips=y,
#     color_values=y,
#     color_function_name="Label",
#     path_html="mapper_cidids2017_raw.html",
#     X=X_scaled,
#     lens=lens,
# )

## Aggregation on Raw Input
df['BruteForce'] = df['Label'].eq(11).astype(int)
df['XSS'] = df['Label'].eq(12).astype(int)
df['SQLi'] = df['Label'].eq(13).astype(int)
df['Benign'] = df['Label'].eq(0).astype(int)

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
    brute_force=('BruteForce', 'mean'),
    xss=('XSS', 'sum'),
    sqli=('SQLi', 'sum'),
    benign=('Benign', 'sum'),
)

## Plotting Aggregated Data
brutes = resampled_data[resampled_data['brute_force'] > 0]
xss = resampled_data[resampled_data['xss'] > 0]
sqli = resampled_data[resampled_data['sqli'] > 0]
# plt.figure(figsize=(8, 8))
# plt.plot(resampled_data.index, resampled_data['flow_count'], color='green', linewidth=1, alpha=0.5, label=f'Aggregated Flow')
# plt.scatter(brutes.index, brutes['flow_count'], color='red', s=10, zorder=5, label='Brute Force')
# plt.scatter(xss.index, xss['flow_count'], color='purple', s=10, zorder=5, label='XSS')
# plt.scatter(sqli.index, sqli['flow_count'], color='orange', s=10, zorder=5, label='SQL Injection')
# plt.legend()
# plt.show()

attributes = [
    'flow_duration', 'flow_mean', 'fwd_mean',
    'bwd_mean', 'active_mean', 'idle_mean',
    'flow_bytes', 'flow_packets', 'flow_count'
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.tight_layout(pad=5.0)

for i, attr in enumerate(attributes):
    row, col = divmod(i, 3)
    axes[row, col].plot(resampled_data.index, resampled_data[attr], label=attr, color='blue')
    axes[row, col].scatter(brutes.index, brutes[attr], color='red', s=10, zorder=5, label='Brute Force')
    axes[row, col].scatter(xss.index, xss[attr], color='purple', s=10, zorder=5, label='XSS')
    axes[row, col].scatter(sqli.index, sqli[attr], color='orange', s=10, zorder=5, label='SQL Injection')
    axes[row, col].set_title(attr)
    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel(attr)
    axes[row, col].legend()

for i in range(len(attributes), 9):
    row, col = divmod(i, 3)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

## Mapper on Aggregated Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(resampled_data)

mapper = km.KeplerMapper(verbose=1)
lens1 = mapper.fit_transform(X_scaled, projection='l2norm')
projector = ensemble.IsolationForest(random_state=42)
projector.fit(X_scaled)
lens2 = projector.decision_function(X_scaled)

lens = np.c_[lens1, lens2]

cover = Cover(n_cubes=20, perc_overlap=0.20)
clusterer = DBSCAN(eps=5, min_samples=5)
# clusterer = KMeans(n_clusters=2, random_state=42)

G = mapper.map(
    lens,
    X_scaled,
    cover=cover,
    clusterer=clusterer,
)

_ = mapper.visualize(
    G,
    custom_tooltips=resampled_data['label'].values,
    color_values=resampled_data['label'].values,
    color_function_name="Label",
    path_html="mapper_cicids2017_resample.html",
    X=X_scaled,
    lens=lens,
)

## Takens's Embedding on Multi Aggregated Data
# features = ['flow_duration', 'flow_mean', 'active_mean', 'fwd_mean', 'bwd_mean']
# featured_resample = resampled_data[features]
# mte = TakensEmbedding(time_delay=1, dimension=5)
# embedding = mte.fit_transform(featured_resample)
# embedding = embedding.reshape(241, 5)

## Takens's Embedding on Raw Data
features = [c for c in df.columns if c not in
            ['Source IP',
             'Source Port',
             'Destination IP',
             'Destination Port',
             'Protocol',
             'Timestamp',
             # 'BruteForce',
             # 'XSS',
             # 'SQLi',
             'Benign',
             'Label']]
featured_df = df[features][:50000]
featured_df.replace([np.inf, -np.inf], np.nan, inplace=True)
featured_df.dropna(inplace=True)
mte = TakensEmbedding(time_delay=1, dimension=featured_df.columns.size)
embedding = mte.fit_transform(featured_df)
embedding = embedding.reshape(featured_df.index.size, featured_df.columns.size)

embedding_scaled = scaler.fit_transform(embedding)
lens1 = mapper.fit_transform(embedding_scaled, projection='l2norm')
projector.fit(embedding_scaled)
lens2 = projector.decision_function(embedding_scaled)

plt.scatter(lens1,lens2,alpha=0.3)
plt.xlabel('PCA')
plt.ylabel('IsolationForest')
plt.show()

lens = np.c_[lens1, lens2]

cover = Cover(n_cubes=20, perc_overlap=0.20)
clusterer = DBSCAN(eps=5, min_samples=5)
# clusterer = KMeans(n_clusters=2, random_state=42)

G = mapper.map(
    lens,
    embedding_scaled,
    cover=cover,
    clusterer=clusterer
)

_ = mapper.visualize(
    G,
    custom_tooltips=df['Label'].values,
    color_values=df['Label'].values,
    color_function_name="target",
    path_html="mapper_cicids2017_takens.html",
    X=embedding_scaled,
    lens=lens,
)