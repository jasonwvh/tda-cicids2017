import matplotlib.pyplot as plt
from DMT_tools.example_generating_functions import *
import warnings
from DMT_tools.utils import MergeTree
warnings.filterwarnings("ignore")

# Import the gtda modules
from gtda.time_series import Resampler, SingleTakensEmbedding, SlidingWindow, PermutationEntropy
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
from gtda.pipeline import Pipeline
from gtda.plotting import plot_point_cloud

# Sine Curve Parameters
domain_start = 0
domain_end = 24*np.pi
n_samples = 200
freq = 1/2
noise_level = 0.1

# Construct the periodic signal
xs = np.linspace(domain_start,domain_end,n_samples)
signal0 = np.sin(xs*freq) + noise_level*np.random.random(n_samples)

# Construct the full signal with a jump
signal = np.zeros(len(xs))
signal[:100] = signal0[:100]
signal[100:200] = signal0[100:200] + 4.5

# Plot the result
plt.plot(xs,signal)
plt.title('Synthetic Time Series Data')
plt.show()

dimension = 10
time_delay = 1

embedder = SingleTakensEmbedding(parameters_type='search',dimension=dimension,time_delay=time_delay)
embedded_signal = embedder.fit_transform(signal)

print('Shape of embedded point cloud:', embedded_signal.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca_coords = pca.fit_transform(embedded_signal)

plt.scatter(pca_coords[:,0],pca_coords[:,1])
plt.title('PCA projection of embedded point cloud')
plt.show()

from sklearn.metrics import euclidean_distances

# Get a density score for each point
DistMat = euclidean_distances(embedded_signal)
mean_dist = np.mean(DistMat)
densities = [sum(DistMat[j,:] < mean_dist/5) for j in range(embedded_signal.shape[0])]

# Subsample by density
total_points = 180
cutoff = np.sort(densities)[::-1][total_points]
# idx = np.argsort(densities)[-total_points:]
embedded_signal_subsampled = embedded_signal[densities >= cutoff,:]

pca_coords = pca.fit_transform(embedded_signal_subsampled)

plt.scatter(pca_coords[:,0],pca_coords[:,1],c = list(range(len(pca_coords))))
plt.title('PCA projection of embedded point cloud, \n subsampled by density')
plt.show()

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
plt.plot(xs,signal, c = 'black', alpha = 0.25, label = 'Decorated Merge Tree')

idx = [i for i in node_labels[0] if densities[i] > cutoff]
x_vals = xs[idx]
y_vals = signal[idx]

plt.scatter(xs[sorted(idx)],signal[sorted(idx)], c = 'g', s = 10, label = 'Original')

idx = [i for i in node_labels[1] if densities[i] > cutoff]
x_vals = xs[idx]
y_vals = signal[idx]

plt.scatter(xs[sorted(idx)],signal[sorted(idx)], c = 'b', s = 10, label = 'Invariant')
plt.show()