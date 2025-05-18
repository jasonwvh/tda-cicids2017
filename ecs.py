import gudhi as gd
import numpy as np
import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from matplotlib import pyplot as plt
from eulearning.utils import vectorize_st, codensity
from helpers import load_and_prep_data
import matplotlib.dates as mdates

df = load_and_prep_data()
df['Timestamp'] = pd.to_datetime(df[' Timestamp'])
df.set_index('Timestamp', inplace=True)

resampled_data = df.resample('30s').agg(
    flow_count=(' Timestamp', 'size'),
    segment_label=(' Label', 'sum')
)

time_series = resampled_data['flow_count']

window_size = 3
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

te = SingleTakensEmbedding('search', 5, 5)
embedding = te.fit_transform(smoothed_time_series)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='b', alpha=0.5)
# ax.set_xlabel('x(t)')
# ax.set_ylabel('x(t-τ)')
# ax.set_zlabel('x(t-2τ)')
# ax.set_title("Takens' Embedding (3D)")
# plt.show()

ac = gd.AlphaComplex(embedding)
st = ac.create_simplex_tree()
vec_st = vectorize_st(st)

from eulearning.descriptors import EulerCharacteristicProfile
euler_curve = EulerCharacteristicProfile(resolution=(200,), val_ranges=[(0, 35)], pt_cld=True, normalize=False)
ecc = euler_curve.fit_transform(vec_st)
ecc_range = np.linspace(euler_curve.val_ranges[0][0], euler_curve.val_ranges[0][1], euler_curve.resolution[0])
plt.figure()
plt.plot(ecc_range, ecc)
plt.title('Euler characteristic curve')
plt.show()

embedding_ = te.fit_transform(smoothed_time_series)
codensity_filt = codensity(embedding_)
vec_st2 = vectorize_st(st, filtrations=[codensity_filt])

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=codensity_filt)
plt.colorbar()
plt.title('2-parameter filtration')
plt.show()

from eulearning.descriptors import EulerCharacteristicProfile

euler_profile = EulerCharacteristicProfile(resolution=(25,25), quantiles=[(0, 0.5), (0, 0.5)], pt_cld=True, normalize=False, flatten=False)
ecs = euler_profile.fit_transform(vec_st2)

extent = list(euler_profile.val_ranges[0]) + list(euler_profile.val_ranges[1])
plt.figure()
plt.imshow(ecs, origin='lower', extent=extent, aspect='auto')
plt.colorbar()
plt.title('Euler characteristic surface')
plt.show()

from eulearning.descriptors import RadonTransform

radon_transform = RadonTransform(resolution=(25,25), quantiles=[0.1, 0.5], pt_cld=True, normalize=False, flatten=False)
rdn = radon_transform.fit_transform(vec_st2)
extent = list(radon_transform.val_ranges[0])+list(radon_transform.val_ranges[1])

plt.figure()
plt.imshow(rdn, origin='lower', extent=extent, aspect='auto')
plt.colorbar()
plt.title('Two-parameter Radon transform')
plt.show()