import os
import random

import numpy as np
import tensorflow as tf

### SEEDING ###################################################################

def seeding(seed: int = 0, tf_deterministic: bool = True):
  np.random.seed(seed=seed)
  tf.random.set_seed(seed=seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  if tf_deterministic:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

###############################################################################

### METRICS ###################################################################

def kl_mvn(m0, S0, m1, S1):
  """
  Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
  Also computes KL divergence from a single Gaussian pm,pv to a set
  of Gaussians qm,qv.
    

  From wikipedia
  KL( (m0, S0) || (m1, S1))
       = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                (m1 - m0)^T S1^{-1} (m1 - m0) - N )
  """
  # store inv diag covariance of S1 and diff between means
  N = m0.shape[0]
  iS1 = np.linalg.inv(S1)
  diff = m1 - m0

  # kl is made of three terms
  tr_term   = np.trace(iS1 @ S0)
  det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
  quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
  
  return .5 * (tr_term + det_term + quad_term - N) 

def frob_mvn(m0, S0, m1, S1):
  diff_mean, diff_cov = m1 - m0, S1 - S0
  
  quad_term = diff_mean.T @ np.linalg.inv(S1) @ diff_mean #np.sum( (diff*diff) * iS1, axis=1)
  cov_metric = diff_cov.T @ diff_cov

  return quad_term + np.trace(cov_metric)

def w2_mvn(m0, S0, m1, S1):
  diff_mean = m1 - m0
  
  evalues, evectors = np.linalg.eig(S1)
  S1_sqrt = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)

  S12 = (S1_sqrt @ S0 @ S1_sqrt)
  evalues, evectors = np.linalg.eig(S12)
  S12_sqrt = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)

  cov_metric = S0 + S1 - 2 * S12_sqrt
  
  return np.linalg.norm(diff_mean) + np.trace(cov_metric)

def rmse_mvn(m0, S0, m1, S1):
  return np.sqrt(np.mean(np.square(m0 - m1))), np.sqrt(np.mean(np.square(S0 - S1)))

###############################################################################

### DATAGEN ###################################################################

def sample_multivariate_normal(num_samples, mean, covariance):
  samples = np.random.multivariate_normal(mean, covariance, size=(num_samples,))
  return tf.data.Dataset.from_tensor_slices(samples)

def sample_random_correlation_via_factorloadings(dim, factors):
  W = np.random.randn(dim, factors)
  S = W @ W.T + np.diag(np.random.rand(1, dim))
  S = np.diag(1./np.sqrt(np.diag(S))) @ S @ np.diag(1./np.sqrt(np.diag(S)))
  return S

def sample_random_covariance_via_svd(dim):
  Q = np.random.randn(dim, dim)
  D = np.diag(abs(np.random.randn(dim)) + 0.2)
  return Q @ D @ Q.T

def sample_random_covariance_via_qr(eigenvalues):
  dim = eigenvalues.shape[0]

  M = np.random.rand(dim, dim)
  Q, _ = np.linalg.qr(M, 'reduced') # random orth matrix

  return Q.T @ np.diag(eigenvalues) @ Q

###############################################################################

### PDFS ######################################################################

def multivariate_normal_pdf(x, mean, covariance):
  dim = mean.shape[0]

  pi_factor = (2.0 * np.pi) ** dim
  normalization_factor = 1.0 / np.sqrt(pi_factor * np.linalg.det(covariance))
  value = np.exp(- 0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean))
  
  return normalization_factor * value

def multivariate_normal_nll(x, mean, covariance):
  return - np.log(multivariate_normal_pdf(x, mean, covariance))

def multivariate_normal_nll_gradient(x, mean, covariance):
  dim = mean.shape[0]

  covariance_inv = np.linalg.inv(covariance)
  mean_gradient = - covariance_inv @ (x - mean)
  covariance_gradient = 0.5 * covariance_inv @ (np.eye(dim) - np.outer((x - mean), (x - mean)) @ covariance_inv)

  return mean_gradient, covariance_gradient

###############################################################################

### FILES #####################################################################

def make_subfolders(folder, subfolders):
  for subfolder in subfolders:
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

###############################################################################

### GEOMETRIC #################################################################

def unit_vector(vector):
  """ Returns the unit of the vector. """
  return vector / np.linalg.norm(vector)

def angle_between(vector1, vector2):
  """ Returns the angle in radians between vectors. """
  vector1_u = unit_vector(vector1)
  vector2_u = unit_vector(vector2)
  return np.arccos(np.clip(np.dot(vector1_u, vector2_u), -1.0, 1.0))

def min_angle_between(vector1, vector2):
  """ Returns the minimal angle in radians between vectors. """
  phi = angle_between(vector1, vector2)
  phi[phi > np.pi / 2] -= np.pi
  return phi

def rotation_matrix_2d(theta):
  """ Returns rotation matrix given angle in radians """
  return np.array([[np.cos(theta),-np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

def rotation_matrix_3d_x(theta):
  return np.array([[ 1, 0            , 0            ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def rotation_matrix_3d_y(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0            , 1, 0            ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def rotation_matrix_3d_z(theta):
  return np.array([[ np.cos(theta),-np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta), 0 ],
                   [ 0            , 0            , 1 ]])

def rotation_matrix_around_span(theta, u, v):
  """ Theta in rads, rotate span(u,v) """
  dim = u.shape[0]
  return np.eye(dim) \
    + np.sin(theta) * (np.outer(v, u) - np.outer(u, v)) \
    + (np.cos(theta) - 1) * (np.outer(u, u) + np.outer(v, v))

def rotation_matrix_around_standard_spans(angles):
  """ Rotates nd matrix around n-1 spans defined by standard basis, using n-1 angles """
  dim = len(angles) + 1
  
  u, vs = np.eye(dim)[0], np.eye(dim)[1:]
  rotation_matrices = [rotation_matrix_around_span(angle, u, v) for angle, v in zip(angles, vs)]

  if len(rotation_matrices) == 1:
    return rotation_matrices[0]
  else:
    return np.linalg.multi_dot(rotation_matrices)

###############################################################################