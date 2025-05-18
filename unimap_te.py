import pandas as pd
import seaborn as sns
import umap
from gtda.time_series import SingleTakensEmbedding
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from helpers import load_and_prep_data

df = load_and_prep_data()
df['Timestamp'] = pd.to_datetime(df[' Timestamp'])
df.sort_values('Timestamp', inplace=True)
df.set_index('Timestamp', inplace=True)

resampled_data = df.resample('30s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']

window_size = 3
smoothed_time_series = time_series.rolling(window=window_size).mean().dropna()
aligned_labels_full = resampled_data['segment_label'].loc[smoothed_time_series.index]

d = 10
tau = 3

te = SingleTakensEmbedding('fixed', tau, d)
embedding = te.fit_transform(smoothed_time_series.values.reshape(-1, 1))
offset = (d - 1) * tau
colors = (aligned_labels_full.iloc[offset:] > 0).astype(int)

reducer = umap.UMAP(random_state=42)
scaled_time_series = StandardScaler().fit_transform(embedding)
umap_embed = reducer.fit_transform(scaled_time_series)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_embed[:, 0],
    umap_embed[:, 1],
    c=colors,
    cmap='coolwarm',
    s=10,
    alpha=0.7
)
plt.title('UMAP Embedding of Time Series with Attack Labels')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='BENIGN',
               markerfacecolor=plt.cm.coolwarm(0.), markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='ATTACK',
               markerfacecolor=plt.cm.coolwarm(1.), markersize=10)
]
plt.legend(handles=legend_elements, title='Label')
plt.colorbar(scatter, ticks=[0, 1], label='Label (0: BENIGN, 1: ATTACK)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()