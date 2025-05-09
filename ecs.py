import gudhi as gd
import numpy as np
import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from matplotlib import pyplot as plt
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

ac = gd.AlphaComplex(embedding)			    # Initialize the alpha complex
st = ac.create_simplex_tree() 		# Compute the alpha filtration
vec_st = vectorize_st(st)		    # Vectorize the simplex tree

# Choose your method to compute Euler characteristic curves
## e.g. using quantiles
euler_curve = EulerCharacteristicProfile(resolution=(200,), quantiles=[(0, 0.95)], pt_cld=True, normalize=False)

# Compute Euler curves
ecc = euler_curve.fit_transform(vec_st)

# Plot Euler curves
ecc_range = np.linspace(euler_curve.val_ranges[0][0], euler_curve.val_ranges[0][1], euler_curve.resolution[0])
plt.figure()
plt.plot(ecc_range, ecc)
plt.title('Euler characteristic curve')
plt.show()

ac = gd.AlphaComplex(embedding)			# Initialize the alpha complex
st = ac.create_simplex_tree() 		# Compute the alpha filtration
vec_st = vectorize_st(st)		# Vectorize the simplex tree
embedding_ = np.array([ac.get_point(i) for i in  range(st.num_vertices())]) 	# For technical reasons, computation of the alpha complex may change the order of vertices
codensity_filt = codensity(embedding_)						# Compute codensity of each point cloud
vec_st2 = vectorize_st(st, filtrations=[codensity_filt])		# Vectorize the simplex tree

plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=codensity_filt)
plt.colorbar()
plt.title('ECS')
plt.show()