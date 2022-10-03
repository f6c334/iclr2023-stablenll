import tensorflow as tf

from tr_projections.tensorflow.projection_utils import kl_diagonal_covariance_projection, kl_covariance_projection


class BaseProjectionLayer(object):
  """Base projection layer

  Args:
      mean_bound ([float]): Mean bound, i.e. eps_mean
      covariance_bound ([float]): Covariance bound, i.e. eps_covariance
  """
  
  def __init__(self, 
               mean_bound : float, 
               covariance_bound : float, 
               parallel_projections: int = 32):
    self.mean_bound = mean_bound
    self.covariance_bound = covariance_bound

    self.parallel_projections = parallel_projections
  
  # as in hom noise : top function and no @tf.function with __call__

  
  # @tf.function(jit_compile=True)
  # def batch_projection(self, means: tf.Tensor, old_means: tf.Tensor, covariances: tf.Tensor, old_covariances: tf.Tensor):
  #   return tf.vectorized_map(
  #     lambda elem : self.projection_(*elem),
  #     elems=(means, old_means, covariances, old_covariances),
  #     # this is for map_fn
  #     #parallel_iterations=self.parallel_projections,
  #     #fn_output_signature=(means.dtype, covariances.dtype)
  #   )
  
  
  @tf.function
  def batch_projection(self, means: tf.Tensor, old_means: tf.Tensor, covariances: tf.Tensor, old_covariances: tf.Tensor):
    return tf.map_fn(
      lambda elem : self.projection_(*elem),
      elems=(means, old_means, covariances, old_covariances),
      # this is for map_fn
      parallel_iterations=self.parallel_projections,
      fn_output_signature=(means.dtype, covariances.dtype)
    )
  
  #@tf.function
  def __call__(self, means : tf.Tensor, old_means : tf.Tensor, covariances : tf.Tensor, old_covariances : tf.Tensor):
    """Call projection layer with batches of means and covariances, project each.

    Args:
        means (tf.Tensor): current predicted means
          0R tensor with one scalar (one 1D projection) [] or
          1R tensor with one N-dim tensor (one ND projection) [N] or
          2R tensor with batch of N-dim tensor [B x N]
        old_means (tf.Tensor): old (projected) covariances
          same as means
        covariances (tf.Tensor): [description]
          0R tensor with one scalar (one 1D projection) [] or
          2R tensor with one NxN-dim tensor (one ND projection) [N X N] or
          3R tensor with batch of N-dim tensor [B x N x N]
        old_covariances (tf.Tensor): [description]
          same as covariances

    Returns:
        [tf.Tensor, tf.Tensor]: projected means and covariances of the same shape as the input
    """
    means_shape, covariances_shape = means.shape, covariances.shape
    
    """
    if tf.rank(means) < 1:  # if scalars
      means, old_means = [tf.expand_dims(var, axis=0) for var in [means, old_means]]
      covariances, old_covariances = tf.reshape(covariances, shape=(1, 1)), tf.reshape(old_covariances, shape=(1, 1))
    """

    means, old_means, covariances, old_covariances = tf.cond(   # if no batches are given
      tf.rank(means) < 2,
      true_fn=lambda : [tf.expand_dims(var, axis=0) for var in [means, old_means, covariances, old_covariances]],
      false_fn=lambda : [means, old_means, covariances, old_covariances],
    )

    # batch execute projections for each (mean, old_mean, covariance, old_covariance) tuple
    proj_means, proj_covariances = self.batch_projection(means, old_means, covariances, old_covariances)

    return tf.reshape(proj_means, shape=means_shape), tf.reshape(proj_covariances, shape=covariances_shape)

  @tf.function
  def projection_(self, mean : tf.Tensor, old_mean : tf.Tensor, covariance : tf.Tensor, old_covariance : tf.Tensor):
    """Project one mean and covariance given its old projected values.

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        covariance (tf.Tensor): current estimated covariance [N x N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor, tf.Tensor]: projected mean and covariance of the same shape as the input
    """
    # calculate mean and cov distance
    mean_distance = self.mean_distance_(mean, old_mean, old_covariance)
    covariance_distance = self.covariance_distance_(covariance, old_covariance)

    # project mean if trust region violated
    if mean_distance > self.mean_bound:
      proj_mean = self.mean_projection_(mean, old_mean, mean_distance)
    else:
      proj_mean = mean
    
    # project variance if trust region violated
    if covariance_distance > self.covariance_bound:
      proj_covariance = self.covariance_projection_(covariance, old_covariance, covariance_distance)
    else:
      proj_covariance = covariance
    
    return proj_mean, proj_covariance
  
  def mean_distance_(self, mean : tf.Tensor, old_mean : tf.Tensor, old_covariance : tf.Tensor):
    """Mahalanobis distance between two means (scaled by the old projected covariance)

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor]: scalar tensor representing the mahalanobis distance
    """
    difference = tf.expand_dims(old_mean - mean, axis=0)
    
    old_covariance_inv = tf.linalg.inv(old_covariance)
    
    mahalanobis_distance = tf.matmul(difference, tf.matmul(old_covariance_inv, tf.transpose(difference)))
    return tf.squeeze(mahalanobis_distance)

  def mean_projection_(self, mean : tf.Tensor, old_mean : tf.Tensor, mean_distance: tf.Tensor):
    """Mahalanobis projection of the mean according to Eq. 6 of (Otto 2021)

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        mean_distance (tf.Tensor): scalar mahalanobis distance
          0D scalar tensor []

    Returns:
        [tf.Tensor]: projected mean, same shape as input [N]
    """
    omega = tf.sqrt(mean_distance / self.mean_bound) - 1.0

    return (mean + omega * old_mean) / (1.0 + omega)
  
  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):
    """Covariance distance respective each projection according to Sec. 4.1, 4.2 of (Otto 2021)

    Args:
        covariance (tf.Tensor): current estimated covariance [N x N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor]: scalar tensor representing the respective layer projection
    """
    raise NotImplementedError("Use a subclass, not the base projection.")

  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
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
  
  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):
    difference = old_covariance - covariance
    cov_metric = tf.matmul(tf.transpose(difference), difference)
    
    return tf.linalg.trace(cov_metric)
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor): 
    eta = tf.sqrt(covariance_distance / self.covariance_bound) - 1.0

    return (covariance + eta * old_covariance) / (1.0 + eta)


class W2ProjectionLayer(BaseProjectionLayer):
  """W2 projection layer, implementing
    - Wasserstein covariance distance, assuming commutative covariances
    - Wasserstein projection (Eq. 9)

    Applicable to full covariance matrices.
  """
  
  def __init__(self, *args, **kwargs):
    super(W2ProjectionLayer, self).__init__(*args, **kwargs)

  def covariance_distance_(self, sqrt_covariance : tf.Tensor, sqrt_old_covariance : tf.Tensor):
    dim = tf.shape(sqrt_covariance)[0]
    identity = tf.eye(dim, dtype=sqrt_covariance.dtype)

    covariance = tf.matmul(sqrt_covariance, sqrt_covariance)
    sqrt_old_covariance_inv = tf.linalg.inv(sqrt_old_covariance)
    old_covariance_inverse = tf.linalg.inv(tf.matmul(sqrt_old_covariance, sqrt_old_covariance))
    
    c = tf.matmul(old_covariance_inverse, covariance)
    d = tf.matmul(sqrt_old_covariance_inv, sqrt_covariance)

    return tf.linalg.trace(identity + c - 2 * d)

  def covariance_projection_(self, sqrt_covariance : tf.Tensor, sqrt_old_covariance : tf.Tensor, covariance_distance: tf.Tensor):    
    eta = tf.sqrt(covariance_distance / self.covariance_bound) - 1.0
    
    return (sqrt_covariance + eta * sqrt_old_covariance) / (1.0 + eta)


class DiagonalKLProjectionLayer(BaseProjectionLayer):
  """Kullback-Leibler projection layer, implementing
    - Kullback-Leibler covariance distance
    - Kullback-Leibler projection (Eq. 11), using L-BFGS optimizer

    Applicable to diagonal covariance matrices only.
  """
  
  def __init__(self, *args, **kwargs):
    super(DiagonalKLProjectionLayer, self).__init__(*args, **kwargs)

  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):    
    dim = tf.cast(tf.shape(covariance)[0], dtype=covariance.dtype)
    diag_covariance, diag_old_covariance = tf.linalg.diag_part(covariance), tf.linalg.diag_part(old_covariance)

    trace_term = tf.reduce_sum((1.0 / diag_old_covariance) * diag_covariance)
    log_term = tf.math.log(tf.reduce_prod(diag_old_covariance) / tf.reduce_prod(diag_covariance))

    return trace_term - dim + log_term
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
    eps = tf.reshape(self.covariance_bound, shape=(1,))
    diag_covariance_, diag_old_covariance_ = tf.linalg.diag_part(covariance), tf.linalg.diag_part(old_covariance)

    # cast to float64 for precision during optimization
    diag_covariance_, diag_old_covariance_ = tf.cast(diag_covariance_, dtype=tf.float64), tf.cast(diag_old_covariance_, dtype=tf.float64)
    eps = tf.cast(eps, dtype=tf.float64)
    
    diag_projected_covariance = kl_diagonal_covariance_projection(diag_covariance_, diag_old_covariance_, eps)
    
    # cast back to previous dtype
    diag_projected_covariance = tf.cast(diag_projected_covariance, dtype=covariance.dtype)
    
    return tf.linalg.diag(diag_projected_covariance)


class DiagonalKLProjectionLayer_(BaseProjectionLayer):
  """Kullback-Leibler projection layer, implementing
    - Kullback-Leibler covariance distance
    - Kullback-Leibler projection (Eq. 11), using L-BFGS optimizer

    Applicable to diagonal covariance matrices only.
  """
  
  def __init__(self, *args, **kwargs):
    super(DiagonalKLProjectionLayer_, self).__init__(*args, **kwargs)

  def mean_distance_(self, mean : tf.Tensor, old_mean : tf.Tensor, old_covariance : tf.Tensor):
    """Mahalanobis distance between two means (scaled by the old projected covariance)

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor]: scalar tensor representing the mahalanobis distance
    """
    difference = old_mean - mean
    old_covariance_inv = 1.0 / old_covariance
    mahalanobis_distance = tf.reduce_sum(tf.square(difference) * old_covariance_inv)
    return mahalanobis_distance
  
  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):    
    dim = tf.cast(tf.shape(covariance)[0], dtype=covariance.dtype)

    trace_term = tf.reduce_sum((1.0 / old_covariance) * covariance)
    log_term = tf.math.log(tf.reduce_prod(old_covariance) / tf.reduce_prod(covariance))

    return trace_term - dim + log_term
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
    eps = tf.reshape(self.covariance_bound, shape=(1,))
    
    # cast to float64 for precision during optimization
    diag_covariance_, diag_old_covariance_ = tf.cast(covariance, dtype=tf.float64), tf.cast(old_covariance, dtype=tf.float64)
    eps = tf.cast(eps, dtype=tf.float64)
    
    diag_projected_covariance = kl_diagonal_covariance_projection(diag_covariance_, diag_old_covariance_, eps)
    
    # cast back to previous dtype
    diag_projected_covariance = tf.cast(diag_projected_covariance, dtype=covariance.dtype)
    
    return diag_projected_covariance
  
class KLProjectionLayer(BaseProjectionLayer):
  """Kullback-Leibler projection layer, implementing
    - Kullback-Leibler covariance distance
    - Kullback-Leibler projection (Eq. 11), using L-BFGS optimizer

    Applicable to full covariance matrices only.
  """
  
  def __init__(self, *args, **kwargs):
    super(KLProjectionLayer, self).__init__(*args, **kwargs)
  
  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):    
    dim = tf.cast(tf.shape(covariance)[0], dtype=covariance.dtype)
    
    trace_term = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(old_covariance), covariance))
    
    logdet_old = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(old_covariance)) + 1e-25))
    logdet_new = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(covariance)) + 1e-25))

    return 0.5 * (trace_term - dim + logdet_old - logdet_new)
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
    eps = tf.cast(tf.reshape(self.covariance_bound, shape=(1,)), dtype=covariance.dtype)
    force_projection = tf.cast(tf.reshape(False, shape=(1,)), dtype=covariance.dtype)
    return kl_covariance_projection(covariance, old_covariance, eps, force_projection)


class KLProjectionLayer_(BaseProjectionLayer):
  """Kullback-Leibler projection layer, implementing
    - Kullback-Leibler covariance distance
    - Kullback-Leibler projection (Eq. 11), using L-BFGS optimizer

    Applicable to full covariance matrices only.
  """
  
  def __init__(self, *args, **kwargs):
    super(KLProjectionLayer, self).__init__(*args, **kwargs)
  
  def covariance_distance_(self, covariance : tf.Tensor, old_covariance : tf.Tensor):    
    dim = tf.cast(tf.shape(covariance)[0], dtype=covariance.dtype)
    
    trace_term = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(old_covariance), covariance))
    
    logdet_old = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(old_covariance)) + 1e-25))
    logdet_new = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(covariance)) + 1e-25))

    return 0.5 * (trace_term - dim + logdet_old - logdet_new)
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
    eps = tf.cast(tf.reshape(self.covariance_bound, shape=(1,)), dtype=covariance.dtype)
    force_projection = tf.cast(tf.reshape(False, shape=(1,)), dtype=covariance.dtype)
    return kl_covariance_projection(covariance, old_covariance, eps, force_projection)


class ForceProjectionLayer(BaseProjectionLayer):
  
  def __init__(self, *args, **kwargs):
    super(ForceProjectionLayer, self).__init__(*args, **kwargs)

  def projection_(self, mean : tf.Tensor, old_mean : tf.Tensor, covariance : tf.Tensor, old_covariance : tf.Tensor):
    """Project one mean and covariance given its old projected values.

    Args:
        mean (tf.Tensor): current estimated mean [N]
        old_mean (tf.Tensor): old (projected) mean [N]
        covariance (tf.Tensor): current estimated covariance [N x N]
        old_covariance (tf.Tensor): old (projected) covariance [N x N]

    Returns:
        [tf.Tensor, tf.Tensor]: projected mean and covariance of the same shape as the input
    """
    # calculate mean and cov distance
    mean_distance = self.mean_distance_(mean, old_mean, old_covariance)
    covariance_distance = self.covariance_distance_(covariance, old_covariance)

    # project mean if trust region violated
    if mean_distance > 0.0:
      proj_mean = self.mean_projection_(mean, old_mean, mean_distance)
    else:
      proj_mean = mean
    
    # project variance if trust region violated
    if covariance_distance > 0.0:
      proj_covariance = self.covariance_projection_(covariance, old_covariance, covariance_distance)
    else:
      proj_covariance = covariance
    
    return proj_mean, proj_covariance


class FrobForceProjectionLayer(ForceProjectionLayer, FrobProjectionLayer):
  
  def __init__(self, *args, **kwargs):
    super(FrobForceProjectionLayer, self).__init__(*args, **kwargs)


class W2ForceProjectionLayer(ForceProjectionLayer, W2ProjectionLayer):
  
  def __init__(self, *args, **kwargs):
    super(W2ForceProjectionLayer, self).__init__(*args, **kwargs)


class KLForceProjectionLayer(ForceProjectionLayer, KLProjectionLayer):
  # Note: Not working properly yet...
  
  def __init__(self, *args, **kwargs):
    super(KLForceProjectionLayer, self).__init__(*args, **kwargs)
  
  def covariance_projection_(self, covariance : tf.Tensor, old_covariance : tf.Tensor, covariance_distance: tf.Tensor):
    eps = tf.cast(tf.reshape(self.covariance_bound, shape=(1,)), dtype=covariance.dtype)
    force_projection = tf.cast(tf.reshape(True, shape=(1,)), dtype=covariance.dtype)
    return kl_covariance_projection(covariance, old_covariance, eps, force_projection)

  """
  #@tf.function
  def batch_projection(self, means: tf.Tensor, old_means: tf.Tensor, covariances: tf.Tensor, old_covariances: tf.Tensor):
    return tf.map_fn(
      lambda elem : self.projection_(*elem),
      elems=(means, old_means, covariances, old_covariances),
      # this is for map_fn
      parallel_iterations=self.parallel_projections,
      fn_output_signature=(means.dtype, covariances.dtype)
    )
  """