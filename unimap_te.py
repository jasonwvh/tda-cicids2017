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

d = 10
tau = 3

te = SingleTakensEmbedding('fixed', tau, d)
embedding = te.fit_transform(smoothed_time_series)

reducer = umap.UMAP()
scaled_time_series = StandardScaler().fit_transform(embedding)
umap_embed = reducer.fit_transform(scaled_time_series)

embedding_times = smoothed_time_series.index[(d - 1) * tau : (d - 1) * tau + len(embedding)]
labels_for_embedding = resampled_data.loc[embedding_times, 'segment_label'].values
colors = [sns.color_palette()[0] if x == 0 else sns.color_palette()[1] for x in labels_for_embedding]

plt.scatter(
    umap_embed[:, 0],
    umap_embed[:, 1],
    c=colors
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=24)
plt.show()