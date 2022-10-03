from turtle import distance
from typing import List, Tuple, Callable

import pickle
import re
import os
import random

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from sklearn import model_selection

###############################################################################
### SEEDING ###################################################################
def seeding(seed: int = 0, tf_deterministic: bool = True):
  tf.keras.utils.set_random_seed(seed)
  if tf_deterministic:
    tf.config.experimental.enable_op_determinism()
  
  # np.random.seed(seed=seed)
  # tf.random.set_seed(seed=seed)
  # random.seed(seed)
  # os.environ['PYTHONHASHSEED'] = str(seed)
  # if tf_deterministic:
  #   os.environ['TF_DETERMINISTIC_OPS'] = '1'


###############################################################################
### DATA NORMALIZATION ########################################################
def normalize(X, mean=None, std=None, epsilon=1e-10):
  if mean is None or std is None:
    mean, std = X.mean(axis=0), X.std(axis=0)
    std += epsilon
  return (X - mean) / std, mean, std


def denormalize(X, mean=None, std=None):
  if type(X) is tuple:  # mean, std denormalization
    mu, var = X
    return mu * std + mean, var * (std[..., tf.newaxis]**2)
  else:
    return X * std + mean


def split_normalize_data(X, Y, normalize_targets=True, test_size=0.3, seed=0):
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

  X_train_normalized, X_train_mean, X_train_std = normalize(X_train)
  Y_train_normalized, Y_train_mean, Y_train_std = normalize(Y_train)
  Y_train_normalized = Y_train_normalized if normalize_targets else Y_train

  X_test_normalized, _, _ = normalize(X_test, X_train_mean, X_train_std)
  if normalize_targets:
    Y_test_normalized, _, _ = normalize(Y_test, Y_train_mean, Y_train_std)
  else:
    Y_test_normalized = Y_test

  train_data, test_data = (X_train_normalized, Y_train_normalized), (X_test_normalized, Y_test_normalized)
  normalization_data = (X_train_mean, X_train_std, Y_train_mean, Y_train_std)

  return train_data, test_data, normalization_data


###############################################################################
### LOSSES ####################################################################
def gauss_nll(x: tf.Tensor, mean: tf.Tensor, covariance_cholesky: tf.Tensor, batch_reduce=tf.reduce_mean) -> tf.Tensor:
  """Computes the multivariate Gauss negative log likelihood for a batch of data.

  Args:
      x (tf.Tensor): batch of data [B x D]
      mean (tf.Tensor): batch of means [B x D]
      covariance_cholesky (tf.Tensor): batch of decomposed covariances [B x D x D]

  Returns:
      tf.Tensor: Mean Gauss NLL of input [1]
  """
  covariance = tf.linalg.matmul(covariance_cholesky, covariance_cholesky, transpose_b=True)

  dim = tf.cast(tf.shape(x)[-1], x.dtype)
  const_term = dim * tf.math.log(2.0 * np.pi)
  log_term = tf.math.log(tf.linalg.det(covariance))
  x_diff = tf.expand_dims(x - mean, axis=-1)
  x_term = tf.linalg.matmul(tf.linalg.matmul(x_diff, tf.linalg.inv(covariance), transpose_a=True), x_diff)
  x_term = tf.squeeze(x_term, axis=(-1, -2))

  lp = -0.5 * (const_term + log_term + x_term)
  nll_ = -batch_reduce(tf.reduce_sum(lp, axis=-1))

  distribution = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=covariance_cholesky)
  nll = -batch_reduce(tf.reduce_sum(distribution.log_prob(x), axis=-1))

  return nll


def gaussian_nll(x: tf.Tensor, mean: tf.Tensor, covariance: tf.Tensor, batch_reduce=tf.reduce_mean) -> tf.Tensor:
  """Computes the multivariate Gauss negative log likelihood for a batch of data.

  Args:
      x (tf.Tensor): batch of data [... x D]
      mean (tf.Tensor): batch of means [... x D]
      covariance (tf.Tensor): batch of covariances [... x D x D]

  Returns:
      tf.Tensor: Gaussian NLL for each input
  """
  dim = tf.cast(tf.shape(x)[-1], x.dtype)

  x_diff = tf.expand_dims(x - mean, axis=-1)
  xmean_term = tf.linalg.matmul(tf.linalg.matmul(x_diff, tf.linalg.inv(covariance), transpose_a=True), x_diff)
  xmean_term = tf.squeeze(xmean_term, axis=(-1, -2))

  log_prob = -0.5 * (dim * tf.math.log(2.0 * tf.cast(np.pi, x.dtype)) + tf.math.log(tf.linalg.det(covariance)) + xmean_term)

  return -batch_reduce(tf.reduce_sum(log_prob, axis=-1))


def kl_divergence(mean: tf.Tensor, covariance_cholesky: tf.Tensor, other_mean: tf.Tensor,
                  other_covariance_cholesky: tf.Tensor) -> tf.Tensor:
  """Computes the mean Kullback-Leibler divergence between a batch of multivariate Gaussians defined by mean/covariance.

  Computes E [ D_KL (p_i || q_i) ], 
  where 
    p_i(x | mean_i, covariance_cholesky_i), 
    q_i(x | other_mean_i, other_covariance_cholesky_i)

  Args:
      mean (tf.Tensor): batch of means of p [B x D]
      covariance_cholesky (tf.Tensor): batch of decomposed covariances of p [B x D x D]
      other_mean (tf.Tensor): batch of means of q [B x D]
      other_covariance_cholesky (tf.Tensor): batch of decomposed covariances of q [B x D x D]

  Returns:
      tf.Tensor: mean KL divergence of batch of input distributions
  """
  current_dist = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=covariance_cholesky)
  other_dist = tfp.distributions.MultivariateNormalTriL(loc=other_mean, scale_tril=other_covariance_cholesky)
  return tf.reduce_mean(current_dist.kl_divergence(other_dist))


def mahalanobis_distance(x: tf.Tensor, y: tf.Tensor, A_inv: tf.Tensor) -> tf.Tensor:
  difference = tf.expand_dims(x - y, axis=-1)
  mahalanobis = tf.matmul(difference, tf.matmul(A_inv, difference), transpose_a=True)
  return tf.squeeze(mahalanobis, axis=[-2, -1])


def wasserstein_distance(A: tf.Tensor, B: tf.Tensor, M_inv: tf.Tensor):  # metric space M to be used
  sqrt_B = tf.linalg.sqrtm(B)

  scaled_A = tf.linalg.matmul(sqrt_B, tf.linalg.matmul(A, sqrt_B))
  inner_term = A + B - 2 * tf.linalg.sqrtm(scaled_A)

  return tf.linalg.trace(tf.linalg.matmul(M_inv, inner_term))


def wasserstein_distance_commutative(A: tf.Tensor, B: tf.Tensor, M_inv: tf.Tensor):  # metric space M to be used
  sqrt_A = tf.linalg.sqrtm(A)
  sqrt_B = tf.linalg.sqrtm(B)

  inner_term = A + B - 2 * tf.linalg.matmul(sqrt_B, sqrt_A)

  return tf.linalg.trace(tf.linalg.matmul(M_inv, inner_term))


def frobenius_distance(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
  return tf.linalg.trace(tf.linalg.matmul(A - B, A - B, transpose_a=True))


def kl_distance(A: tf.Tensor, B: tf.Tensor):
  d = tf.cast(tf.shape(A)[-1], dtype=A.dtype)
  log_term = tf.math.log(tf.linalg.det(B) / tf.linalg.det(A))
  trace_term = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(B), A))

  return 0.5 * (log_term + trace_term - d)


###############################################################################
### DATA MANIPULATION #########################################################
def nested_flatten(tensor_list: List[tf.Tensor]) -> tf.Tensor:
  """Flatten a list of variable sized tensors into one tensor.

  Args:
      tensor_list (List[tf.Tensor]): list of tensors, can have any shape

  Returns:
      tf.Tensor: flat tensor containing all elements of input list
  """
  flattened_gradients = [tf.reshape(gradient, shape=(-1,)) for gradient in tensor_list]
  return tf.concat(flattened_gradients, axis=-1)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter


def nested_unflatten(tensor_list: List[tf.Tensor], flat_tensor: tf.Tensor) -> List[tf.Tensor]:
  """Unflattens flat tensor into list of tensor of variable shapes defined by reference tensor_list.

  Args:
      tensor_list (List[tf.Tensor]): reference list of tensors the flat tensor should be shaped into
      flat_tensor (tf.Tensor): rank 1 tensor with as many elements as defined by tensor_list

  Returns:
      List[tf.Tensor]: list of tensors, with shapes of individual tensors as in reference list
  """
  weight_shapes = [w.get_shape() for w in tensor_list]
  params_per_layer = [tf.reduce_prod(w_shape) for w_shape in weight_shapes]
  gradients_per_layer = tf.split(flat_tensor, num_or_size_splits=params_per_layer, axis=0)

  return [tf.reshape(w, shape=w_shape) for w, w_shape in zip(gradients_per_layer, weight_shapes)]


def nested_assign(tensor_list: List[tf.Tensor], new_tensor_list: List[tf.Tensor]) -> List[tf.Tensor]:
  """Unflattens flat tensor into list of tensor of variable shapes defined by reference tensor_list.

  Args:
      tensor_list (List[tf.Tensor]): reference list of tensors the flat tensor should be shaped into
      flat_tensor (tf.Tensor): rank 1 tensor with as many elements as defined by tensor_list

  Returns:
      List[tf.Tensor]: list of tensors, with shapes of individual tensors as in reference list
  """
  for tensor, new_tensor in zip(tensor_list, tf.identity_n(new_tensor_list)):
    tensor.assign(new_tensor)


###############################################################################
### OTHER #####################################################################
def conjugate_gradient(Ax: Callable[[tf.Tensor], tf.Tensor],
                       b: tf.Tensor,
                       x0: tf.Tensor,
                       max_iterations: int = 200,
                       residual_tolerance: float = 1e-05,
                       epsilon: float = 1e-08) -> Tuple[tf.Tensor, tf.Tensor]:
  """Solves equation system Ax = b for x using the conjugate gradient algorithm.

  Note: Here Ax is a callable returning the matrix vector product given x,
    therefore, no need to compute A or its inverse explicitely.

  Args:
      Ax (Callable[[tf.Tensor], tf.Tensor]): Callable returning Ax given x [D -> D]
      b (tf.Tensor): flat tensor representing solution space [D]
      x0 (tf.Tensor): flat tensor representing initial solution [D]
      max_iterations (int, optional): maximum iterations of the CG algorithm. Defaults to 200.
      residual_tolerance (float, optional): residual tolerance (rr^T) at which to stop the CG algorithm. Defaults to 1e-05.
      epsilon (float, optional): numerical stabilizer constant. Defaults to 1e-08.

  Returns:
      Tuple[tf.Tensor, tf.Tensor]: returns (x, r) such that Ax = b with residual error of r
  """
  x = tf.identity(x0)
  r = b - Ax(x)
  p, r_dot_old = tf.identity(r), tf.tensordot(r, r, 1)

  def conjugate_gradient_step(x, r, p, r_dot_old):
    z = Ax(p)

    alpha = r_dot_old / (tf.tensordot(p, z, 1) + epsilon)
    x += alpha * p
    r -= alpha * z

    r_dot_new = tf.tensordot(r, r, 1)
    beta = r_dot_new / (r_dot_old + epsilon)
    r_dot_old = r_dot_new

    p = r + beta * p

    return x, r, p, r_dot_old

  x, r, p, r_dot_old = tf.while_loop(cond=lambda x, r, p, r_dot_old: tf.greater(r_dot_old, residual_tolerance),
                                     body=conjugate_gradient_step,
                                     loop_vars=[x, r, p, r_dot_old],
                                     maximum_iterations=max_iterations)

  return x, r_dot_old


def linesearch(loss_fn: Callable[[tf.Tensor], tf.Tensor],
               constraint_fn: Callable[[tf.Tensor], tf.Tensor],
               x: tf.Tensor,
               full_step: tf.Tensor,
               delta: float,
               max_iterations: int = 200,
               backtrack_coefficient: float = 0.6) -> Tuple[tf.Tensor, tf.Tensor]:
  coefficient = 1.0
  x_new = tf.identity(x + coefficient * full_step)
  loss, constraint = loss_fn(x_new), constraint_fn(x_new)

  def linesearch_step(coefficient, x_new, loss, constraint):
    coefficient *= backtrack_coefficient
    x_new = x + coefficient * full_step
    loss, constraint = loss_fn(x_new), constraint_fn(x_new)
    return coefficient, x_new, loss, constraint

  coefficient, x_new, loss, constraint = tf.while_loop(
    cond=lambda coefficient, x_new, loss, constraint: tf.greater(constraint, delta),
    body=linesearch_step,
    loop_vars=[coefficient, x_new, loss, constraint],
    maximum_iterations=max_iterations)

  return tf.cond(constraint > delta,
                 true_fn=lambda: (x, tf.zeros_like(constraint)),
                 false_fn=lambda: (x_new, constraint))


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


def sample_batch_mvn(mean: np.ndarray, cov: np.ndarray, size: "tuple | int" = ()) -> np.ndarray:
  """
    Batch sample multivariate normal distribution.

    Arguments:

        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)

    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)

    It is not required that ``mean`` and ``cov`` have the same shape
    prefix, only that they are broadcastable against each other.
  """
  mean = np.asarray(mean)
  cov = np.asarray(cov)
  size = (size,) if isinstance(size, int) else tuple(size)
  shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
  X = np.random.standard_normal((*shape, 1))
  L = np.linalg.cholesky(cov)
  return (L @ X).reshape(shape) + mean


def glob_re(pattern, strings):
  return filter(re.compile(pattern).match, strings)


def load_history(file_path):
  return pickle.load(open(file_path, 'rb'))


def get_curves_from_key(logdir, key):
  file_paths = glob_re(r'.*\.pkl', os.listdir(logdir))
  histories = [load_history(os.path.join(logdir, file_path)) for file_path in file_paths]
  chosen_key = list(map(lambda h: h[key], histories))
  
  idx = range(len(chosen_key[0]))
  curves = list(map(lambda h: np.asarray([idx, h]).T, chosen_key))

  return curves