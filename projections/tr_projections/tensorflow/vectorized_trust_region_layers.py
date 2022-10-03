import tensorflow as tf

from tr_projections.tensorflow.projection_utils import kl_covariance_projection
from tr_projections.tensorflow.vectorized_projection_utils import kl_covariance_projection as kl_covariance_projection_
from tr_projections.tensorflow.vectorized_projection_utils import kl_covariance_projection_test

class BaseProjectionLayer(object):
  """Base projection layer

  Args:
      mean_bound ([float]): Mean bound, i.e. eps_mean
      covariance_bound ([float]): Covariance bound, i.e. eps_covariance
  """
  
  def __init__(self, 
               mean_bound : float, 
               covariance_bound : float):
    self.mean_bound_ = mean_bound
    self.covariance_bound_ = covariance_bound
    self.mean_bound = tf.constant(value=mean_bound, dtype=tf.keras.backend.floatx())
    self.covariance_bound = tf.constant(value=covariance_bound, dtype=tf.keras.backend.floatx())
  
  @tf.function
  def __call__(self, means : tf.Tensor, old_means : tf.Tensor, covariances : tf.Tensor, old_covariances : tf.Tensor):
    """Call projection layer with batches of means and covariances, project each.

    Args:
        means (tf.Tensor): current predicted means
          2R tensor with batch of N-dim tensor [B x N]
        old_means (tf.Tensor): old (projected) covariances
          same as means
        covariances (tf.Tensor): [description]
          3R tensor with batch of N-dim tensor [B x N x N]
        old_covariances (tf.Tensor): [description]
          same as covariances

    Returns:
        [tf.Tensor, tf.Tensor]: projected means and covariances of the same shape as the input
    """
    means, old_means, covariances, old_covariances = tf.cond(   # if no batches are given
      tf.rank(means) < 2,
      true_fn=lambda : [tf.expand_dims(var, axis=0) for var in [means, old_means, covariances, old_covariances]],
      false_fn=lambda : [means, old_means, covariances, old_covariances],
    )
    
    # compute mean, covariance distances
    means_distances = self.means_distances(means, old_means, old_covariances)
    covariances_distances = self.covariances_distances(covariances, old_covariances)

    # project mean if needed
    idx_means_to_project = tf.where(means_distances > self.mean_bound)

    means_updates = self.means_projections(
      means=tf.gather_nd(means, idx_means_to_project),
      old_means=tf.gather_nd(old_means, idx_means_to_project),
      means_distances=tf.gather_nd(means_distances, idx_means_to_project)
    )
    proj_means = tf.tensor_scatter_nd_update(
      means, idx_means_to_project,
      updates=means_updates
    )

    # project covariance if needed
    idx_covariances_to_project = tf.where(covariances_distances > self.covariance_bound)
    
    covariances_updates = self.covariances_projections(
      covariances=tf.gather_nd(covariances, idx_covariances_to_project),
      old_covariances=tf.gather_nd(old_covariances, idx_covariances_to_project),
      covariances_distances=tf.gather_nd(covariances_distances, idx_covariances_to_project)
    )
    proj_covariances = tf.tensor_scatter_nd_update(
      covariances, idx_covariances_to_project,
      updates=covariances_updates
    )

    return tf.reshape(proj_means, shape=tf.shape(means)), tf.reshape(proj_covariances, shape=tf.shape(covariances))
  
  def means_distances(self, means : tf.Tensor, old_means : tf.Tensor, old_covariances : tf.Tensor):
    """Mahalanobis distance between two means (scaled by the old projected covariance)

    Args:
        mean (tf.Tensor): current estimated mean [B x N]
        old_mean (tf.Tensor): old (projected) mean [B x N]
        old_covariance (tf.Tensor): old (projected) covariance [B x N x N]

    Returns:
        [tf.Tensor]: scalar tensor representing the mahalanobis distances [B]
    """
    differences = tf.expand_dims(old_means - means, axis=-1)
    old_covariances_inv = tf.linalg.inv(old_covariances)
    
    mahalanobis = tf.matmul(differences, tf.matmul(old_covariances_inv, differences), transpose_a=True)
    return tf.squeeze(mahalanobis, axis=[-2, -1])

  def means_projections(self, means : tf.Tensor, old_means : tf.Tensor, means_distances: tf.Tensor):
    """Mahalanobis projection of the mean according to Eq. 6 of (Otto 2021)

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        mean_distance (tf.Tensor): scalar mahalanobis distance
          0D scalar tensor []

    Returns:
        [tf.Tensor]: projected mean, same shape as input [N]
    """
    omegas = tf.sqrt(means_distances / self.mean_bound) - 1.0    
    omegas = omegas[:, tf.newaxis]

    return (means + omegas * old_means) / (1.0 + omegas)
  
  def covariances_distances(self, covariances : tf.Tensor, old_covariances : tf.Tensor):
    """Covariance distance respective each projection according to Sec. 4.1, 4.2 of (Otto 2021)

    Args:
        covariance (tf.Tensor): current estimated covariance [N x N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor]: scalar tensor representing the respective layer projection
    """
    raise NotImplementedError("Use a subclass, not the base projection.")

  def covariances_projections(self, covariances : tf.Tensor, old_covariances : tf.Tensor, covariances_distances: tf.Tensor):
    """Covariance projection respective each projection according to Eq. 7, 9, 11 of (Otto 2021)

    Args:
        covariance (tf.Tensor): current estimated covariance [N x N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]
        covariance_distance (tf.Tensor): scalar distance given respective norm []

    Returns:
        [tf.Tensor]: projected covariance, same shape as input [N x N]
    """
    raise NotImplementedError("Use a subclass, not the base projection.")


class FrobProjectionLayer(BaseProjectionLayer):
  """Frobenius projection layer, implementing
    - Frobenius covariance distance
    - Frobenius projection (Eq. 7)

    Applicable to full covariance matrices.
  """

  def __init__(self, *args, **kwargs):
    super(FrobProjectionLayer, self).__init__(*args, **kwargs)
  
  def covariances_distances(self, covariances : tf.Tensor, old_covariances : tf.Tensor):
    differences = old_covariances - covariances
    cov_metric = tf.matmul(differences, differences, transpose_a=True)
    
    return tf.linalg.trace(cov_metric)

  def covariances_projections(self, covariances : tf.Tensor, old_covariances : tf.Tensor, covariances_distances: tf.Tensor):
    eta = tf.sqrt(covariances_distances / self.covariance_bound) - 1.0
    eta = eta[:, tf.newaxis, tf.newaxis]

    return (covariances + eta * old_covariances) / (1.0 + eta)


class W2ProjectionLayer(BaseProjectionLayer):
  """W2 projection layer, implementing
    - Wasserstein covariance distance, assuming commutative covariances
    - Wasserstein projection (Eq. 9)

    Applicable to full covariance matrices.
  """
  
  def __init__(self, *args, **kwargs):
    super(W2ProjectionLayer, self).__init__(*args, **kwargs)

  def covariances_distances(self, covariances : tf.Tensor, old_covariances : tf.Tensor):
    sqrt_covariances = tf.linalg.sqrtm(covariances)
    sqrt_old_covariances = tf.linalg.sqrtm(old_covariances)

    old_covariances_inv = tf.linalg.inv(old_covariances)
    sqrt_old_covariances_inv = tf.linalg.inv(sqrt_old_covariances)

    c = tf.matmul(old_covariances_inv, covariances)
    d = tf.matmul(sqrt_old_covariances_inv, sqrt_covariances)
    # d = tf.matmul(old_covariances_inv, tf.linalg.sqrtm(tf.linalg.matmul(sqrt_old_covariances, tf.linalg.matmul(covariances, sqrt_old_covariances))))

    identity = tf.eye(tf.shape(covariances)[-1], dtype=covariances.dtype)
    return tf.linalg.trace(identity + c - 2 * d)

  def covariances_projections(self, covariances : tf.Tensor, old_covariances : tf.Tensor, covariances_distances: tf.Tensor):
    eta = tf.sqrt(covariances_distances / self.covariance_bound) - 1.0
    eta = eta[:, tf.newaxis, tf.newaxis]

    sqrt_covariances = tf.linalg.sqrtm(covariances)
    sqrt_old_covariances = tf.linalg.sqrtm(old_covariances)

    sqrt_proj_covariances = (sqrt_covariances + eta * sqrt_old_covariances) / (1.0 + eta)
    return tf.matmul(sqrt_proj_covariances, sqrt_proj_covariances)






class KLProjectionLayer(BaseProjectionLayer):
  """Kullback-Leibler projection layer, implementing
    - Kullback-Leibler covariance distance
    - Kullback-Leibler projection (Eq. 11), using L-BFGS optimizer

    Applicable to full covariance matrices.
  """
  
  def __init__(self, *args, **kwargs):
    super(KLProjectionLayer, self).__init__(*args, **kwargs)
  
#   def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
#     eps = tf.cast(tf.reshape(self.covariance_bound, shape=(1,)), dtype=covariance.dtype)
#     force_projection = tf.cast(tf.reshape(False, shape=(1,)), dtype=covariance.dtype)
#     return kl_covariance_projection(covariance, old_covariance, eps, force_projection)

  def covariances_distances(self, covariances : tf.Tensor, old_covariances : tf.Tensor):
    dim = covariances.shape[1]
    old_covariances_inv = tf.linalg.inv(old_covariances)

    trace_terms = tf.linalg.trace(tf.linalg.matmul(old_covariances_inv, covariances))
    
    logdets_old = tf.math.log(tf.linalg.det(old_covariances) + 1e-15)
    logdets_new = tf.math.log(tf.linalg.det(covariances) + 1e-15)
    # logdet_old = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(old_covariance)) + 1e-25))
    # logdet_new = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(covariance)) + 1e-25))

    return 0.5 * (logdets_old - logdets_new + trace_terms - dim)

  def covariances_projections(self, covariances : tf.Tensor, old_covariances : tf.Tensor, covariances_distances: tf.Tensor):
    eps = tf.cast(tf.reshape(self.covariance_bound, shape=(1,)), dtype=covariances.dtype)
    force_projection = tf.cast(tf.reshape(False, shape=(1,)), dtype=covariances.dtype)
    
    eps_ = tf.expand_dims(tf.repeat(eps, repeats=tf.shape(covariances)[0]), axis=-1)
    force_projection_ = tf.expand_dims(tf.repeat(force_projection, repeats=tf.shape(covariances)[0]), axis=-1)

    # proj_covariances = tf.map_fn(
    #   lambda elem : kl_covariance_projection(*elem),
    #   elems=(covariances, old_covariances, eps_, force_projection_),
    #   fn_output_signature=covariances.dtype,
    #   parallel_iterations=10
    # )
    # tf.print(proj_covariances)
    # return kl_covariance_projection_(covariances, old_covariances, self.covariance_bound, 0.0)
    return kl_covariance_projection_test(covariances, old_covariances, self.covariance_bound, 0.0)