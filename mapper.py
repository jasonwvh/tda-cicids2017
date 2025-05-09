import kmapper as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gtda.time_series import SingleTakensEmbedding
from sklearn import ensemble, cluster
from sklearn.preprocessing import StandardScaler
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

    df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding='ISO-8859-1', usecols=columns_to_keep, low_memory=False)
    df.replace('Infinity', -1, inplace=True)
    df[" Source IP"] = df[" Source IP"].apply(lambda x: float(str(x).replace(".", "")))
    df[" Destination IP"] = df[" Destination IP"].apply(lambda x: float(str(x).replace(".", "")))
    df = df.dropna()

    df[" Label"] = df[" Label"].map({"BENIGN": 0, "Web Attack  Brute Force": 1, "Web Attack  XSS": 1, "Web Attack  Sql Injection": 1})
    return df

df = load_and_prep_data()

df['Timestamp'] = pd.to_datetime(df[' Timestamp'])
df.sort_values('Timestamp', inplace=True)
df.set_index('Timestamp', inplace=True)

resampled_data = df.resample('60s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']
segment_labels = resampled_data['segment_label'].values

plt.figure(figsize=(12, 4))
plt.plot(time_series.index, time_series.values, color='b', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Number of Flows per Second')
plt.title('Time Series: Network Flows per Second')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

te = SingleTakensEmbedding('search', 1, 3)
embedding = te.fit_transform(time_series)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='b', s=10, alpha=0.5)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t-τ)')
ax.set_title("Takens' Embedding (2D)")
plt.show()

features = [c for c in df.columns if c not in
            [' Source IP',
             ' Source Port',
             ' Destination IP',
             ' Destination Port',
             ' Protocol',
             ' Timestamp',
             ' Label']]

X = embedding
y = segment_labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
eps = np.mean(distances[:, 4])
print(f"Computed eps for DBSCAN: {eps}")

projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(X_scaled)
lens1 = projector.decision_function(X_scaled)

mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X_scaled, projection="knn_distance_5")

lens = np.c_[lens1, lens2]

G = mapper.map(
    lens,
    X_scaled,
    cover=km.Cover(n_cubes=20, perc_overlap=0.20),
    clusterer=cluster.DBSCAN(eps=5, min_samples=5)
)

print(f"Number of nodes: {len(G['nodes'])}")
print(f"Number of edges: {sum([len(values) for key, values in G['links'].items()])}")

_ = mapper.visualize(
    G,
    custom_tooltips=y,
    color_values=y,
    color_function_name="target",
    path_html="output.html",
    X=X_scaled,
    X_names=list(df[features].columns),
    lens=lens,
    lens_names=["IsolationForest", "KNN-distance 5"],
    title="Detecting network anomaly with topology"
)