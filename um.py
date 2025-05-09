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

resampled_data = df.resample('60s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']
te = SingleTakensEmbedding('fixed', 1, 3)
embedding = te.fit_transform(time_series)

reducer = umap.UMAP()
scaled_time_series = StandardScaler().fit_transform(embedding)
umap_embed = reducer.fit_transform(scaled_time_series)

plt.scatter(
    umap_embed[:, 0],
    umap_embed[:, 1],
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=24)
plt.show()