import tensorflow as tf
import tensorflow_probability as tfp


@tf.function(jit_compile=True)
@tf.custom_gradient
def kl_diagonal_covariance_projection(covariance: tf.Tensor, old_covariance: tf.Tensor, eps: tf.Tensor):
  """
  Calculating covariance projection based on Eq. 11 of (Otto, 2021)
  and the gradient as in the Appendix

  Args:
      covariance (tf.Tensor): current estimated covariance
        diagonal matrix [D]
      old_covariance (tf.Tensor): old (projected) covariance
        diagonal matrix [D]
      eps (tf.Tensor): covariance bound
        1D float tensor, one value of shape (not scalar) [1]
  """
  ### PREPROCESSING ###########################################################
  dim, dtype = tf.cast(tf.shape(covariance)[0], dtype=covariance.dtype), covariance.dtype

  omega_offset = 1.0

  old_precision = 1.0 / old_covariance
  old_cholesky_precision = tf.sqrt(old_precision)

  target_precision = 1.0 / covariance
  old_logdet = - 2.0 * tf.reduce_sum(tf.math.log(old_cholesky_precision + 1e-25))

  kl_const_part = old_logdet - dim  
  

  ### LBFGS OPTIMIZATION ######################################################
  @tf.function(jit_compile=True)
  def dual(eta):
    """Implements dual of optimization problem (Otto, 2021 Eq. 17) w.r.t. eta

    Args:
        eta ([tf.Tensor]): lagrangian multiplier eta [1]

    Returns:
        [(float, tf.Tensor)]: dual function and gradient 
          (Eq. 18 onwards, with simplification)
    """
    if eta < 0.0:
      eta = tf.constant([0.0], dtype=dtype)
    
    new_precision = (eta * old_precision + target_precision) / (eta + omega_offset)
    
    new_covariance = 1.0 / new_precision
    new_cholesky_covariance = tf.sqrt(new_covariance)
    new_logdet = 2.0 * tf.reduce_sum(tf.math.log(new_cholesky_covariance + 1e-25))

    dual = eta * eps - 0.5 * eta * old_logdet + 0.5 * (eta + omega_offset) * new_logdet
    
    trace_term = tf.reduce_sum(tf.square(old_cholesky_precision * new_cholesky_covariance))
    kl = 0.5 * (kl_const_part - new_logdet + trace_term)    
    gradient = (eps - kl)

    return tf.reduce_sum(dual), gradient

  # using vectorizable tfp::lbfgs optimizer for parallelization
  start = tf.constant([1.0], dtype=dtype)
  optim_results = tfp.optimizer.lbfgs_minimize(
    dual,
    initial_position=start,
    num_correction_pairs=10,
    max_iterations=300,
    parallel_iterations=8,
    tolerance=1e-8
  )

  eta_opt = tf.clip_by_value(optim_results.position[0], 0.0, 1e12)

  # projection with optimal eta
  projected_precision = (eta_opt * old_precision + target_precision) / (eta_opt + omega_offset)
  projected_covariance = 1.0 / projected_precision
  
  
  ### GRADIENT OF PROJECTION ##################################################
  @tf.function(jit_compile=True)
  def _custom_gradient(upstream):
    if eta_opt == 0.0:
      deta_dQ_target = tf.zeros(shape=(dim), dtype=dtype)
    else:
      dQ_deta = (omega_offset * old_precision - target_precision) / (eta_opt + omega_offset)
      f2_dQ = projected_covariance * (tf.ones(shape=(dim), dtype=dtype) - old_precision * projected_covariance)
      c = - 1.0 / tf.reduce_sum(f2_dQ * dQ_deta)
      deta_dQ_target = c * f2_dQ
    
    eo = omega_offset + eta_opt
    eo_squared = eo * eo
    dQ_deta = (omega_offset * old_precision - target_precision) / eo_squared

    d_Q = - projected_covariance * upstream * projected_covariance
    d_eta = tf.reduce_sum(d_Q * dQ_deta)

    d_Q_target = d_eta * deta_dQ_target + d_Q / eo

    d_cov_target = - target_precision * d_Q_target * target_precision

    # note: only need the cov gradient, the rest doesn't matter
    return [d_cov_target, tf.zeros_like(old_covariance), tf.zeros_like(eps)]

  return projected_covariance, _custom_gradient

@tf.function(jit_compile=True)
@tf.custom_gradient
def kl_covariance_projection(covariance: tf.Tensor, old_covariance: tf.Tensor, eps: tf.Tensor, force_projection: tf.Tensor):
  """
  Calculating covariance projection based on Eq. 11 of (Otto, 2021)
  and the gradient as in the Appendix

  Args:
      covariance (tf.Tensor): current estimated covariance
        diagonal matrix [D]
      old_covariance (tf.Tensor): old (projected) covariance
        diagonal matrix [D]
      eps (tf.Tensor): covariance bound
        1D float tensor, one value of shape (not scalar) [1]
  """
  ### PREPROCESSING ###########################################################
  dim, dtype = tf.cast(tf.shape(covariance)[0], covariance.dtype), covariance.dtype

  omega_offset = 1.0

  old_precision = tf.linalg.inv(old_covariance)
  old_cholesky_precision = tf.linalg.cholesky(old_precision)

  target_precision = tf.linalg.inv(covariance)
  old_logdet = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(old_cholesky_precision) + 1e-25))

  kl_const_part = old_logdet - dim
  

  ### LBFGS OPTIMIZATION ######################################################
  @tf.function(jit_compile=True)
  def dual(eta):
    """Implements dual of optimization problem (Otto, 2021 Eq. 17) w.r.t. eta

    Args:
        eta ([tf.Tensor]): lagrangian multiplier eta [1]

    Returns:
        [(float, tf.Tensor)]: dual function and gradient 
          (Eq. 18 onwards, with simplification)
    """
    if eta < 0.0 and force_projection <= 0.0:
      eta = tf.constant([0.0], dtype=dtype)
    
    new_precision = (eta * old_precision + target_precision) / (eta + omega_offset)
    
    new_covariance = tf.linalg.inv(new_precision)
    new_cholesky_covariance = tf.linalg.cholesky(new_covariance)
    new_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(new_cholesky_covariance) + 1e-25))

    dual = eta * eps - 0.5 * eta * old_logdet + 0.5 * (eta + omega_offset) * new_logdet
    
    trace_term = tf.linalg.trace(tf.linalg.matmul(old_precision, new_cholesky_covariance))
    kl = 0.5 * (kl_const_part - new_logdet + trace_term)
    gradient = (eps - kl)

    return tf.reduce_sum(dual), gradient

  # using vectorizable tfp::lbfgs optimizer for parallelization
  start = tf.constant([1.0], dtype=dtype)
  optim_results = tfp.optimizer.lbfgs_minimize(
    dual,
    initial_position=start,
    num_correction_pairs=2,
    max_iterations=50,
    parallel_iterations=10,
    max_line_search_iterations=50,
    tolerance=1e-8
  )

  eta_opt = tf.cond(force_projection <= 0.0, 
                    lambda : tf.clip_by_value(optim_results.position[0], 0.0, 1e12),
                    lambda : tf.clip_by_value(optim_results.position[0], -1e12, 1e12))
  
  # projection with optimal eta
  projected_precision = (eta_opt * old_precision + target_precision) / (eta_opt + omega_offset)
  projected_covariance = tf.linalg.inv(projected_precision)
  

  ### GRADIENT OF PROJECTION ##################################################
  @tf.function(jit_compile=True)
  def _custom_gradient(upstream):
    if eta_opt == 0.0:
      deta_dQ_target = tf.zeros(shape=(dim, dim), dtype=dtype)
    else:
      dQ_deta = (omega_offset * old_precision - target_precision) / (eta_opt + omega_offset)
      f2_dQ = tf.linalg.matmul(projected_covariance, (tf.eye(dim, dtype=dtype) - tf.linalg.matmul(old_precision, projected_covariance)))
      c = - 1.0 / tf.linalg.trace(tf.linalg.matmul(f2_dQ, dQ_deta)) # todo here
      deta_dQ_target = c * f2_dQ
    
    eo_squared = tf.square(omega_offset + eta_opt)
    dQ_deta = (omega_offset * old_precision - target_precision) / eo_squared

    d_Q = - tf.linalg.matmul(tf.linalg.matmul(projected_covariance, upstream), projected_covariance)
    d_eta = tf.linalg.trace(tf.linalg.matmul(d_Q, dQ_deta))

    d_Q_target = d_eta * deta_dQ_target + d_Q / (omega_offset + eta_opt)

    d_cov_target = - tf.linalg.matmul(tf.linalg.matmul(target_precision, d_Q_target), target_precision)

    # note: only need the cov gradient, the rest doesn't matter
    return [d_cov_target, tf.zeros_like(old_covariance), tf.zeros_like(eps), tf.zeros_like(force_projection)]
  
  return projected_covariance, _custom_gradient
