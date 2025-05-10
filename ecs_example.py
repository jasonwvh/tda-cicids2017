import matplotlib.pyplot as plt
from eulearning.datasets import gen_dense_lines_with_noise

X = gen_dense_lines_with_noise(n_pattern=2)

plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()

import numpy as np
import gudhi as gd
from eulearning.utils import vectorize_st

ac = gd.AlphaComplex(X)			# Initialize the alpha complex
st = ac.create_simplex_tree() 		# Compute the alpha filtration
vec_st = vectorize_st(st)		# Vectorize the simplex tree

# from eulearning.descriptors import EulerCharacteristicProfile
#
# # Choose your method to compute Euler characteristic curves
# ## e.g. using quantiles
# euler_curve = EulerCharacteristicProfile(resolution=(200,), quantiles=[(0, 0.95)], pt_cld=True, normalize=False)
#
# ## e.g. using specified bounds
# # euler_curve = EulerCharacteristicProfile(resolution=(200,), val_ranges=[(0, 35)], pt_cld=True, normalize=False)
#
# # Compute Euler curves
# ecc = euler_curve.fit_transform(vec_st)
#
# # Plot Euler curves
# ecc_range = np.linspace(euler_curve.val_ranges[0][0], euler_curve.val_ranges[0][1], euler_curve.resolution[0])
# plt.figure()
# plt.plot(ecc_range, ecc)
# plt.title('Euler characteristic curve')
# plt.show()
#
# from eulearning.descriptors import HybridTransform
#
# # Choose your method to compute hybrid transform
# ## e.g. using specified bounds and a specified kernel:
# kernel = lambda x:np.exp(np.cos(x))
# hyb_trans = HybridTransform(resolution=(200,), val_ranges=[(0,5)], kernel=kernel, pt_cld=True, normalize=False)
#
# ## e.g. using quantiles and a keyword kernel:
# # Available keyword kernels are :
# # 		- 'exp_p' 		: lambda x: np.exp(-np.abs(x)**p)
# # 		- 'wavelet_p' 		: lambda x: x**p * np.exp(-x**p)
# # 		- 'cos_p' 		: lambda x: np.cos(x**p)
# hyb_trans = HybridTransform(resolution=(200,), quantiles=[0.05], kernel_name='wavelet_4', pt_cld=True, normalize=False)
#
# # Compute hybrid transform
# ht = hyb_trans.fit_transform(vec_st)

# # Plot hybrid transform
# ht_range = np.linspace(hyb_trans.val_ranges[0][0], hyb_trans.val_ranges[0][1], hyb_trans.resolution[0])
# plt.figure()
# plt.plot(ht_range, ht)
# plt.title('Hybrid transform')
# plt.show()

from eulearning.utils import codensity

X_ = np.array([ac.get_point(i) for i in  range(st.num_vertices())]) 	# For technical reasons, computation of the alpha complex may change the order of vertices
codensity_filt = codensity(X_)						# Compute codensity of each point cloud
vec_st2 = vectorize_st(st, filtrations=[codensity_filt])		# Vectorize the simplex tree

plt.figure()
plt.scatter(X[:,0], X[:,1], c=codensity_filt)
plt.colorbar()
plt.title('Toy example')
plt.show()

from eulearning.descriptors import EulerCharacteristicProfile

euler_profile = EulerCharacteristicProfile(resolution=(100,100), quantiles=[(0, 0.9), (0.1,0.9)], pt_cld=True, normalize=False, flatten=False)
ecs = euler_profile.fit_transform(vec_st2)

# Plot the result
extent = list(euler_profile.val_ranges[0]) + list(euler_profile.val_ranges[1])
plt.figure()
plt.imshow(ecs, origin='lower', extent=extent, aspect='auto')
plt.colorbar()
plt.title('Euler characteristic surface')
plt.show()

# from eulearning.descriptors import HybridTransform
#
# hybrid_transform2 = HybridTransform(resolution=(100,100), quantiles=[0.3, 0.2], kernel_name='wavelet_4', pt_cld=True, normalize=False, flatten=False)
# multi_ht = hybrid_transform2.fit_transform(vec_st2)
#
# extent = list(hybrid_transform2.val_ranges[0]) + list(hybrid_transform2.val_ranges[1])
# plt.figure()
# plt.imshow(multi_ht, origin='lower', extent=extent, aspect='auto')
# plt.colorbar()
# plt.title('Two-parameter hybrid transform')
# plt.show()
#
# from eulearning.descriptors import RadonTransform
#
# radon_transform = RadonTransform(resolution=(100,100), quantiles=[0.2, 1], pt_cld=True, normalize=False, flatten=False)
# rdn = radon_transform.fit_transform(vec_st2)
# extent = list(radon_transform.val_ranges[0])+list(radon_transform.val_ranges[1])
#
# plt.figure()
# plt.imshow(rdn, origin='lower', extent=extent, aspect='auto')
# plt.colorbar()
# plt.title('Two-parameter Radon transform')
# plt.show()