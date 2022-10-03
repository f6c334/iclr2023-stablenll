import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils.common import sample_random_covariance_via_svd, multivariate_normal_nll, multivariate_normal_nll_gradient, sample_random_covariance_via_qr
from utils.plotting import plot_arrow





def sample_random_pd_2x2_eigfnc(eig_fnc, det, epsilon=1e-10):
  X = sample_random_covariance_via_qr(eig_fnc(det))

  if (np.linalg.det(X) - det) / det > epsilon:
    return sample_random_pd_2x2_eigfnc(eig_fnc, det, epsilon)
  return X 
   

def sample_random_pd_2x2_eigfnc_range(eig_fnc, det_range, samples_det, samples_per_det, check_valid=True):
  min_det, max_det = det_range

  covariances = np.asarray([sample_random_pd_2x2_eigfnc(eig_fnc, det)
                            for _ in range(samples_per_det) 
                            for det in np.linspace(min_det, max_det, samples_det)])
  dets = np.asarray([np.linalg.det(covariance) for covariance in covariances])

  valid_idx = np.logical_and(min_det < dets, dets < max_det)
  return (covariances[valid_idx], dets[valid_idx]) if check_valid else (covariances, dets)



def plot_gradients_vs_det(dxmu, eig_fnc, det_range, samples_det, samples_per_det):
  covariances, dets = sample_random_pd_2x2_eigfnc_range(eig_fnc, det_range, samples_det, samples_per_det)
  grad_norms = np.asarray([np.linalg.norm(multivariate_normal_nll_gradient(np.zeros((2)), dxmu, X)[1]) for X in covariances])

  min_eigvals = np.asarray([np.min(np.linalg.eigvals(X)) for X in covariances])

  fig, ax = plt.subplots()
  plot = ax.scatter(dets, grad_norms, c=min_eigvals, cmap='magma')

  ax.set_yscale('log')

  ax.set_xlabel('det $\Sigma$')
  ax.set_ylabel('||Gradient NLL||')

  cbar = fig.colorbar(plot, aspect=50)
  cbar.set_label('Smallest Eigenvalue')

  fig.set_size_inches(16, 9)
  fig.set_tight_layout('True')
  return fig, ax

def get_eigscale_eig_fnc(eigscale):
  def eig_fnc(det):
    eig1 = eigscale * np.random.rand()
    return np.asarray([eig1, det / eig1])
  return eig_fnc

def get_evendist_eig_fnc():
  def eig_fnc(det):
    eig1 = det * np.random.rand()
    return np.asarray([eig1, det / eig1])
  return eig_fnc

import matplotlib as mpl
plt.style.use('./science.mplstyle')
mpl.rcParams['font.size'] = 40.48

folder = './non-contextual/figures/2d_gradients_plots/'
os.makedirs(folder, exist_ok=True)

fig, ax = plot_gradients_vs_det(dxmu=1.0 * np.ones((2)), eig_fnc=get_eigscale_eig_fnc(0.01), det_range=(1e-6, 100.0), samples_det=1000, samples_per_det=100)
fig.savefig(os.path.join(folder, 'gradients_on_detSigma_onesmalleigen.png'), dpi=300)
fig, ax = plot_gradients_vs_det(dxmu=1.0 * np.ones((2)), eig_fnc=get_eigscale_eig_fnc(1.0), det_range=(1e-6, 100.0), samples_det=1000, samples_per_det=100)
fig.savefig(os.path.join(folder, 'gradients_on_detSigma_randomeigen.png'), dpi=300)
fig, ax = plot_gradients_vs_det(dxmu=1.0 * np.ones((2)), eig_fnc=get_evendist_eig_fnc(), det_range=(1e-6, 100.0), samples_det=1000, samples_per_det=100)
fig.savefig(os.path.join(folder, 'gradients_on_detSigma_randomeigen_smalldxmu.png'), dpi=300)

plt.savefig(os.path.join(folder, 'gradients_on_detSigma_evenrandomeigen.png'), dpi=1200)
plt.show()
quit()







#dxmu = 0.0 * np.ones((2))
eig_fnc = get_evendist_eig_fnc()
det_range = (1e-6, 10)
samples_det = 100
samples_per_det = 100


covariances, dets = sample_random_pd_2x2_eigfnc_range(eig_fnc, det_range, samples_det, samples_per_det)
dxmus = 1.0 * np.random.rand(covariances.shape[0], 2) - 1.0

gradients = [multivariate_normal_nll_gradient(np.zeros(2), dxmu, X) for dxmu, X in zip(dxmus, covariances)]
covariance_gradients = np.asarray([gradient[1] for gradient in gradients])


alpha = 1.0
new_covariances = covariances - alpha * covariance_gradients

eigvals_covariances, eigvecs_covariances = np.linalg.eig(covariances)
eigvals_new_covariances, eigvecs_new_covariances = np.linalg.eig(new_covariances)
#eigvals_diff_cov, eigvecs_diff_cov = np.linalg.eig(dxmus[..., None] * dxmus[:, None] - covariances)
eigvals_diff_cov, eigvecs_diff_cov = np.linalg.eig([np.linalg.inv(X) @ (np.outer(dxmu, dxmu)) for dxmu, X in zip(dxmus, covariances)])


# B x D x D

def batch_rotations(vecsA, vecsB):
  scalar_products = np.einsum('ij,ij->i', vecsA, vecsB)
  cos_phi = scalar_products / (np.linalg.norm(vecsA, axis=1) * np.linalg.norm(vecsB, axis=1))
  phi = np.rad2deg(np.arccos(cos_phi))

  phi[phi > 90] = 180 - phi[phi > 90]

  return phi

rotations1 = batch_rotations(eigvecs_covariances[:, 0], eigvecs_new_covariances[:, 0])
rotations2 = batch_rotations(eigvecs_covariances[:, 0], eigvecs_new_covariances[:, 1])
rotations_no = np.minimum(rotations1, rotations2)

rotations1 = batch_rotations(eigvecs_covariances[:, 0], eigvecs_diff_cov[:, 0])
rotations2 = batch_rotations(eigvecs_covariances[:, 0], eigvecs_diff_cov[:, 1])
rotations_di = np.minimum(rotations1, rotations2)

fig, ax = plt.subplots()
plot = ax.scatter(rotations_di, rotations_no, c=np.min(eigvals_diff_cov, axis=1))#, c=min_eigvals, cmap='magma')

#ax.set_yscale('log')

ax.set_xlabel('det $\Sigma$')
ax.set_ylabel('Rotation')

cbar = fig.colorbar(plot, aspect=50)
cbar.set_label('Smallest Eigenvalue')

fig.set_size_inches(16, 9)
fig.set_tight_layout('True')

plt.show()
















"""
##### GRID PLOT OF COVARIANCE EIGENVECTORS

dxmu = 0.5 * np.ones((2))
eig_fnc = get_evendist_eig_fnc()
det_range = (0.5, 1.5)
samples_det = 5
samples_per_det = 5


covariances, dets = sample_random_pd_2x2_eigfnc_range(eig_fnc, det_range, samples_det, samples_per_det, check_valid=False)
gradients = [multivariate_normal_nll_gradient(np.zeros(2), dxmu, X) for X in covariances]
covariance_gradients = np.asarray([gradient[1] for gradient in gradients])


alpha = 1.0
new_covariances = covariances - alpha * covariance_gradients

eigvals_covariances, eigvecs_covariances = np.linalg.eig(covariances)
eigvals_new_covariances, eigvecs_new_covariances = np.linalg.eig(new_covariances)

def plot_covariance_eigen_2x2(covariance, **kwargs):
  vals, vecs = np.linalg.eig(covariance)

  for val, vec in zip(vals, vecs):
    plot_arrow(np.zeros_like(vec), val * vec, **kwargs)

fig, axes = plt.subplots(samples_per_det, samples_det)

for i in range(samples_per_det):
  for j in range(samples_det):
    plot_covariance_eigen_2x2(covariances[i + j * samples_per_det], ax=axes[i, j], color='red')
    plot_covariance_eigen_2x2(new_covariances[i + j * samples_per_det], ax=axes[i, j], color='blue')

    axes[i, j].set_aspect('equal')

#axes.set_aspect('equal')


#ax.set_xscale('symlog')
#ax.set_yscale('symlog')


#ax.quiver(*origin, *point, color=['r'], angles='xy', scale_units='xy', scale=1)

#ax.quiver(origin, V[:,1], color=['b'], scale=21)

#def plot_eigenvectors_2x2():

plt.show()
#grad_norms = np.asarray([np.linalg.norm(multivariate_normal_nll_gradient(np.zeros((2)), dxmu, X)[1]) for X in covariances])

"""






































"""
### 3D PLOTS OF RANDOM EIGS DEPENDING ON THINGYS

# plot NLL


# five dof
eigscale, det_range = 1.0, (1e-6, 1.0)
samples_det, samples_per_det = 100, 100

covariances, dets = sample_random_pd_2x2_eigfnc_range(eigscale, det_range, samples_det, samples_per_det)
dxmus = 2.0 * np.random.rand(covariances.shape[0], 2) - 1.0

zs = np.asarray([np.linalg.norm(multivariate_normal_nll_gradient(np.zeros((2)), dxmu, covariance)[1]) for covariance, dxmu in zip(covariances, dxmus)])




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

xs = dets
ys = np.asarray([np.linalg.norm(X) for X in dxmus])
min_eigvals = np.asarray([np.min(np.linalg.eigvals(X)) for X in covariances])



ax.scatter(xs, ys, np.log10(zs), c=min_eigvals, cmap='magma')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('log dNLL')


# fig.add_subplot(1, 2, 2, projection='3d')
fig.set_size_inches(16, 9)
fig.set_tight_layout('True')

plt.show()




# generate random pd matrices with certain determinants / eigenvalues

# compute, given a certain delta_xmu, the LL gradient norm (dLL/dSigma)

"""


















# conjecture:
# X 1) larger determinants = larger eigenvalues == small gradients
# X 2) smaller determinants = smaller eigenvalues == large gradients
#   3) somehow there must be a relation between delta_xmu_i and the eigenvalues
#         - such that the gradient becomes larger/smaller
#         - for example, if the delta_xmu in dimension 1 is large and eigenvalue 1 is large and vice versa
#         - or, if correlation between dimensions is large then delta_xmu in each dim is more influencial etc.
#   4) could we use the determinant lemma somehow? (det(A + UWV^T) = det(W^-1 + V^T A^-1 U) det(W) det(A))
#   5) signed determinant is the scaling factor of a matrix, thus this could be useful to describe why (formally)
#   6) nll sgd variance lines up eigenvectors, and this might be a local minimum... ?

# results:
#   1) (1) and (2) are correct, but there are more cases in which the gradient is large (these might be only with super high elements in the matrix in the first place)
#   2) in fact, the smallest eigenvalue determines how large the gradient will be (see second plot smalleigen)
#   3) if dxmu is larger, then eigenvalues basically define gradient size, IF dxmu is small, then large eigenvalues will be centering gradient around zero