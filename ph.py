import pandas as pd
import matplotlib.pyplot as plt
from gtda.diagrams import PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding

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
plt.plot(time_series.index, time_series.values, color='b', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Number of Flows per Second')
plt.title('Time Series: Network Flows per Second')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

te = SingleTakensEmbedding('fixed', 1, 3)
embedding = te.fit_transform(time_series)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='b', s=10, alpha=0.5)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t-τ)')
ax.set_zlabel('x(t-2τ)')
ax.set_title("Takens' Embedding (3D)")
plt.show()

# segment_labels = resampled_data['segment_label']
# plt.plot(segment_labels.index, segment_labels.values, color='b', linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Number of Flows per Second')
# plt.title('Time Series: Network Flows per Second')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# te_label = SingleTakensEmbedding('fixed', 1, 2)
# embedding_label = te_label.fit_transform(segment_labels)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(embedding_label[:, 0], embedding_label[:, 1], embedding_label[:, 2], c='b', s=10, alpha=0.5)
# ax.set_xlabel('x(t)')
# ax.set_ylabel('x(t-τ)')
# ax.set_zlabel('x(t-2τ)')
# ax.set_title("Takens' Embedding (3D)")
# plt.show()

ph = VietorisRipsPersistence(homology_dimensions=[0, 1])
embedding_reshaped = embedding[None, :, :]
diagrams = ph.fit_transform(embedding_reshaped)
fig = plt.figure(figsize=(10, 8))
for i, dgm in enumerate(diagrams):
    if len(dgm) > 0:
        plt.scatter(dgm[:, 0], dgm[:, 1], label=f'H{i}', alpha=0.6)
        plt.plot([0, max(dgm[:, 1])], [0, max(dgm[:, 1])], 'k--', alpha=0.5)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('Persistence Diagrams')
plt.legend()
plt.grid(True)
plt.show()

pl = PersistenceLandscape()
landscapes = pl.fit_transform(diagrams)
fig = plt.figure(figsize=(10, 8))
for i in range(landscapes.shape[1]):
    plt.plot(landscapes[0, i], label=f'Landscape {i}')
plt.xlabel('Filtration value')
plt.ylabel('Landscape value')
plt.legend()
plt.title('Persistence Landscapes')
plt.show()
