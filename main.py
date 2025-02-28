import kmapper as km
import numpy as np
import pandas as pd
from sklearn import ensemble, cluster

def load_and_prep_data():
    columns_to_keep = [
        ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp',
        ' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
        ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
        ' Fwd IAT Min', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
        'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
        ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
    ]

    df = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding='ISO-8859-1', usecols=columns_to_keep)
    df.replace('Infinity', -1, inplace=True)
    df[" Source IP"] = df[" Source IP"].apply(lambda x: float(str(x).replace(".", "")))
    df[" Destination IP"] = df[" Destination IP"].apply(lambda x: float(str(x).replace(".", "")))

    df = df.dropna()

    df[" Label"] = df[" Label"].map({"BENIGN": 0, "Web Attack  Brute Force": 1, "Web Attack  XSS": 1, "Web Attack  Sql Injection": 1})

    return df

df = load_and_prep_data()

features = [c for c in df.columns if c not in
            [' Source IP',
             ' Source Port',
             ' Destination IP',
             ' Destination Port',
             ' Protocol',
             ' Timestamp',
             ' Label']]

X = np.array(df[features])
X[np.isinf(X)] = np.nan
X = np.nan_to_num(X, nan=-1)
y = np.array(df[' Label'])

projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(X)
lens1 = projector.decision_function(X)

mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="knn_distance_5")

lens = np.c_[lens1, lens2]

G = mapper.map(
    lens,
    X,
    cover = km.Cover(n_cubes=20,
                     perc_overlap=.20),
    clusterer=cluster.KMeans(n_clusters=15))

print(f"num nodes: {len(G['nodes'])}")
print(f"num edges: {sum([len(values) for key, values in G['links'].items()])}")

_ = mapper.visualize(
    G,
    custom_tooltips=y,
    color_values=y,
    color_function_name="target",
    path_html="output.html",
    X=X,
    X_names=list(df[features].columns),
    lens=lens,
    lens_names=["IsolationForest", "KNN-distance 5"],
    title="Detecting network anomaly with Isolation Forest and Nearest Neighbor Distance"
)