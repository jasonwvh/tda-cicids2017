import numpy as np
import pandas as pd
import umap
from gtda.time_series import SingleTakensEmbedding
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from DMT_tools.utils import MergeTree
from helpers import load_and_prep_data
from sklearn.decomposition import PCA

df = load_and_prep_data()
df['Timestamp'] = pd.to_datetime(df[' Timestamp'])
df.set_index('Timestamp', inplace=True)

resampled_data = df.resample('30s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']

window_size = 4
smoothed_time_series = time_series.rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series.values, color='b', linewidth=1, alpha=0.5, label='Original')
plt.plot(smoothed_time_series.index, smoothed_time_series.values, color='g', linewidth=2, label=f'Smoothed (Window={window_size})')

attack_points = resampled_data[resampled_data['segment_label'] > 0]
plt.scatter(attack_points.index, attack_points['flow_count'], color='r', s=10, zorder=5, label='Attack Segment')

ax = plt.gca()
date_form = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xlabel('Time')
plt.ylabel('Number of Flows')
plt.title('Time Series: Network Flows (Original and Smoothed)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

smoothed_time_series = smoothed_time_series.dropna()

# te = SingleTakensEmbedding('fixed', 7, 2)
# embedding = te.fit_transform(smoothed_time_series)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)
# ax.plot(embedding[:, 0], embedding[:, 1], c='b', alpha=0.5)
# ax.set_xlabel('x(t)')
# ax.set_ylabel('x(t-τ)')
# ax.set_title("Takens' Embedding (2D)")
# plt.show()

te = SingleTakensEmbedding('search', 2, 3)
embedding = te.fit_transform(smoothed_time_series)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='b', alpha=0.5)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t-τ)')
ax.set_zlabel('x(t-2τ)')
ax.set_title("Takens' Embedding (3D)")
plt.show()

# ---
xs = smoothed_time_series.index
signal = smoothed_time_series.values
embedded_signal = embedding

#---
aligned_labels_full = resampled_data['segment_label'].loc[smoothed_time_series.index]

d = 10
tau = 3

te = SingleTakensEmbedding('fixed', tau, d)
embedding = te.fit_transform(smoothed_time_series.values.reshape(-1, 1))
offset = (d - 1) * tau
colors = (aligned_labels_full.iloc[offset:] > 0).astype(int)

reducer = umap.UMAP(random_state=42)
scaled_time_series = StandardScaler().fit_transform(embedding)

DistMat = euclidean_distances(embedded_signal)
mean_dist = np.mean(DistMat)
densities = [sum(DistMat[j,:] < mean_dist/5) for j in range(embedded_signal.shape[0])]

total_points = 180
cutoff = np.sort(densities)[::-1][total_points]
# idx = np.argsort(densities)[-total_points:]
embedded_signal_subsampled = embedded_signal[densities >= cutoff,:]

umap_coords = reducer.fit_transform(embedded_signal_subsampled)

plt.scatter(umap_coords[:,0],umap_coords[:,1],c = list(range(len(umap_coords))))
plt.title('UMAP projection of embedded point cloud, \n subsampled by density')
plt.show()

# ---
# pca = PCA(n_components = 2)
# pca_coords = pca.fit_transform(embedded_signal)
#
# plt.scatter(pca_coords[:,0],pca_coords[:,1])
# plt.title('PCA projection of embedded point cloud')
# plt.show()
#
# DistMat = euclidean_distances(embedded_signal)
# mean_dist = np.mean(DistMat)
# densities = [sum(DistMat[j,:] < mean_dist/5) for j in range(embedded_signal.shape[0])]
#
# total_points = 180
# cutoff = np.sort(densities)[::-1][total_points]
# # idx = np.argsort(densities)[-total_points:]
# embedded_signal_subsampled = embedded_signal[densities >= cutoff,:]
#
# pca_coords = pca.fit_transform(embedded_signal_subsampled)
#
# plt.scatter(pca_coords[:,0],pca_coords[:,1],c = list(range(len(pca_coords))))
# plt.title('PCA projection of embedded point cloud, \n subsampled by density')
# plt.show()

#---

MT = MergeTree(pointCloud = embedded_signal_subsampled)
MT.fit_barcode(degree=1)

tree_thresh = 0.1
barcode_thresh = 0.1

MT.draw_decorated(tree_thresh,barcode_thresh)

num_bars = 2

node_labels = {}

barcode = MT.barcode
barcode_lengths = [bar[1] - bar[0] for bar in barcode]
barcode_idx_sorted_by_length = np.argsort(barcode_lengths)[::-1]

leaf_barcode = MT.leaf_barcode

for i in range(num_bars):

    labels = []

    idx = barcode_idx_sorted_by_length[i]
    bar = barcode[idx]

    for leaf, bc in leaf_barcode.items():
        if list(bar) in bc:
            labels.append(leaf)

    node_labels[i] = labels

plt.figure(figsize = (7,5))
plt.plot(xs,signal, c = 'black', alpha = 0.25)

idx = [i for i in node_labels[0] if densities[i] > cutoff]
x_vals = xs[idx]
y_vals = signal[idx]
plt.scatter(xs[sorted(idx)],signal[sorted(idx)], c = 'g', s = 10, label='Original')

idx = [i for i in node_labels[1] if densities[i] > cutoff]
plt.scatter(xs[sorted(idx)],signal[sorted(idx)], c = 'b', s = 10, label='Invariant')

plt.scatter(attack_points.index, attack_points['flow_count'], color='r', s=10, zorder=5, label='Attacks')

ax = plt.gca()
date_form = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xlabel('Time')
plt.ylabel('Number of Flows')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()