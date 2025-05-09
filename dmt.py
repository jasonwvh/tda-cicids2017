import gudhi as gd
import numpy as np
import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

from DMT_tools.utils import MergeTree
from eulearning.datasets import gen_dense_lines_with_noise
from eulearning.utils import vectorize_st, codensity
from eulearning.descriptors import EulerCharacteristicProfile
from helpers import load_and_prep_data

df = load_and_prep_data()
df['Timestamp'] = pd.to_datetime(df[' Timestamp'])
df.sort_values('Timestamp', inplace=True)
df.set_index('Timestamp', inplace=True)

resampled_data = df.resample('60s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']
te = SingleTakensEmbedding('fixed', 1, 2)
embedding = te.fit_transform(time_series)

pca = PCA(n_components = 2)
pca_coords = pca.fit_transform(embedding)

# plt.scatter(pca_coords[:,0],pca_coords[:,1])
# plt.title('PCA projection of embedded point cloud')
# plt.show()

# Get a density score for each point
DistMat = euclidean_distances(embedding)
mean_dist = np.mean(DistMat)
densities = [sum(DistMat[j,:] < mean_dist/5) for j in range(embedding.shape[0])]

# Subsample by density
total_points = 180
cutoff = np.sort(densities)[::-1][total_points]
# idx = np.argsort(densities)[-total_points:]
embedding_subsampled = embedding[densities >= cutoff,:]

pca_coords = pca.fit_transform(embedding_subsampled)

# plt.scatter(pca_coords[:,0],pca_coords[:,1],c = list(range(len(pca_coords))))
# plt.title('PCA projection of embedded point cloud, \n subsampled by density')
# plt.show()

MT = MergeTree(pointCloud = embedding_subsampled)
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

xs = time_series.index
idx = [i for i in node_labels[0] if densities[i] > cutoff]
x_vals = xs[idx]
y_vals = embedding[idx]

plt.scatter(xs[sorted(idx)],time_series[sorted(idx)], c = 'g', s = 10)

idx = [i for i in node_labels[1] if densities[i] > cutoff]
x_vals = xs[idx]
y_vals = embedding[idx]

plt.scatter(xs[sorted(idx)],time_series[sorted(idx)], c = 'b', s = 10)
plt.show()