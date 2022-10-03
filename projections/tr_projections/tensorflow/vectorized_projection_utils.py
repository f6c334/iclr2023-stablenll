import tensorflow as tf
import tensorflow_probability as tfp

@tf.function(jit_compile=True)
def _optimize(target_precision, old_precision, old_cholesky_precision, old_logdet, kl_const_part, eps, force_projection):
  omega_offset = 1.0

  def dual(eta):
    """Implements dual of optimization problem (Otto, 2021 Eq. 17) w.r.t. eta

    Args:
        eta ([tf.Tensor]): lagrangian multiplier eta [1]

    Returns:
        [(float, tf.Tensor)]: dual function and gradient 
          (Eq. 18 onwards, with simplification)
    """
    if eta < 0.0 and force_projection <= 0.0:
      eta = tf.constant([0.0], dtype=target_precision.dtype)
    
    new_precision = (eta * old_precision + target_precision) / (eta + omega_offset)
    
    new_covariance = tf.linalg.inv(new_precision)
    new_cholesky_covariance = tf.linalg.cholesky(new_covariance)
    new_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(new_cholesky_covariance) + 1e-25))
    # new_logdet = tf.math.log(tf.linalg.det(new_covariance) + 1e-15)

    dual = eta * eps - 0.5 * eta * old_logdet + 0.5 * (eta + omega_offset) * new_logdet
    
    # trace_term = tf.linalg.trace(tf.linalg.matmul(old_precision, new_covariance))
    trace_term = tf.linalg.trace(tf.linalg.matmul(old_precision, new_covariance))
    kl = 0.5 * (kl_const_part - new_logdet + trace_term)
    gradient = (eps - kl)
    
    return tf.reduce_sum(dual), gradient

  # using vectorizable tfp::lbfgs optimizer for parallelization
  start = tf.constant([1.0], dtype=target_precision.dtype)
  optim_results = tfp.optimizer.lbfgs_minimize(
    dual,
    initial_position=start,
    max_iterations=50,
    tolerance=1e-4
  )
  # tf.print(optim_results.converged, optim_results.position[0])
  eta_opt = tf.clip_by_value(optim_results.position[0], 0.0, 1e12)
  return eta_opt


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
  batch_dim, dim = tf.shape(covariance)[0], tf.shape(covariance)[-1]

  omega_offset = 1.0

  old_precision = tf.linalg.inv(old_covariance)
  old_logdet = tf.math.log(tf.linalg.det(old_covariance) + 1e-15)

  target_precision = tf.linalg.inv(covariance)

  kl_const_part = old_logdet - tf.cast(dim, dtype=covariance.dtype)


  ### LBFGS OPTIMIZATION ######################################################

  eps_ = tf.expand_dims(tf.repeat(eps, repeats=tf.shape(covariance)[0]), axis=-1)
  force_projection_ = tf.expand_dims(tf.repeat(force_projection, repeats=tf.shape(covariance)[0]), axis=-1)

  eta_opts = tf.map_fn(
    lambda elem : _optimize(*elem),
    elems=(target_precision, old_precision, tf.linalg.cholesky(old_precision), old_logdet, kl_const_part, eps_, force_projection_),
    fn_output_signature=covariance.dtype,
    parallel_iterations=256
  )

  # eta_opts = tf.vectorized_map(
  #   lambda elem : _optimize(*elem),
  #   elems=(target_precision, old_precision, tf.linalg.cholesky(old_precision), old_logdet, kl_const_part, eps_, force_projection_),
  #   # fn_output_signature=covariance.dtype,
  #   # parallel_iterations=256
  # )

  tf.print('\nISNAN', tf.reduce_any(tf.math.is_nan(eta_opts)))




  def mock_custom_gradient(upstream):
    d_cov_target = upstream * tf.eye(covariance.shape[1], dtype=covariance.dtype)

    # note: only need the cov gradient, the rest doesn't matter
    return d_cov_target, tf.zeros_like(old_covariance), tf.zeros_like(eps), tf.zeros_like(force_projection)

  return covariance, mock_custom_gradient




  

  # ### LBFGS OPTIMIZATION ######################################################
  # @tf.function(jit_compile=True)
  # def dual(eta):
  #   """Implements dual of optimization problem (Otto, 2021 Eq. 17) w.r.t. eta

  #   Args:
  #       eta ([tf.Tensor]): lagrangian multiplier eta [1]

  #   Returns:
  #       [(float, tf.Tensor)]: dual function and gradient 
  #         (Eq. 18 onwards, with simplification)
  #   """
  #   if eta < 0.0 and force_projection <= 0.0:
  #     eta = tf.constant([0.0], dtype=dtype)
    
  #   new_precision = (eta * old_precision + target_precision) / (eta + omega_offset)
    
  #   new_covariance = tf.linalg.inv(new_precision)
  #   new_cholesky_covariance = tf.linalg.cholesky(new_covariance)
  #   new_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(new_cholesky_covariance) + 1e-25))

  #   dual = eta * eps - 0.5 * eta * old_logdet + 0.5 * (eta + omega_offset) * new_logdet
    
  #   trace_term = tf.linalg.trace(tf.linalg.matmul(old_precision, new_cholesky_covariance))
  #   kl = 0.5 * (kl_const_part - new_logdet + trace_term)
  #   gradient = (eps - kl)

  #   return tf.reduce_sum(dual), gradient

  # # using vectorizable tfp::lbfgs optimizer for parallelization
  # start = tf.constant([1.0], dtype=dtype)
  # optim_results = tfp.optimizer.lbfgs_minimize(
  #   dual,
  #   initial_position=start,
  #   num_correction_pairs=2,
  #   max_iterations=20,
  #   parallel_iterations=10,
  #   max_line_search_iterations=50,
  #   tolerance=1e-8
  # )

  # eta_opt = tf.cond(force_projection <= 0.0, 
  #                   lambda : tf.clip_by_value(optim_results.position[0], 0.0, 1e12),
  #                   lambda : tf.clip_by_value(optim_results.position[0], -1e12, 1e12))
  
  # # projection with optimal eta
  # projected_precision = (eta_opt * old_precision + target_precision) / (eta_opt + omega_offset)
  # projected_covariance = tf.linalg.inv(projected_precision)
  

  # ### GRADIENT OF PROJECTION ##################################################
  # @tf.function(jit_compile=True)
  # def _custom_gradient(upstream):
  #   if eta_opt == 0.0:
  #     deta_dQ_target = tf.zeros(shape=(dim, dim), dtype=dtype)
  #   else:
  #     dQ_deta = (omega_offset * old_precision - target_precision) / (eta_opt + omega_offset)
  #     f2_dQ = tf.linalg.matmul(projected_covariance, (tf.eye(dim, dtype=dtype) - tf.linalg.matmul(old_precision, projected_covariance)))
  #     c = - 1.0 / tf.linalg.trace(tf.linalg.matmul(f2_dQ, dQ_deta)) # todo here
  #     deta_dQ_target = c * f2_dQ
    
  #   eo_squared = tf.square(omega_offset + eta_opt)
  #   dQ_deta = (omega_offset * old_precision - target_precision) / eo_squared

  #   d_Q = - tf.linalg.matmul(tf.linalg.matmul(projected_covariance, upstream), projected_covariance)
  #   d_eta = tf.linalg.trace(tf.linalg.matmul(d_Q, dQ_deta))

  #   d_Q_target = d_eta * deta_dQ_target + d_Q / (omega_offset + eta_opt)

  #   d_cov_target = - tf.linalg.matmul(tf.linalg.matmul(target_precision, d_Q_target), target_precision)

  #   # note: only need the cov gradient, the rest doesn't matter
  #   return [d_cov_target, tf.zeros_like(old_covariance), tf.zeros_like(eps), tf.zeros_like(force_projection)]
  
  # return projected_covariance, _custom_gradient





@tf.custom_gradient
def kl_covariance_projection_test(covariance: tf.Tensor, old_covariance: tf.Tensor, eps: tf.Tensor, force_projection: tf.Tensor):
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
  batch_dim, dim = tf.shape(covariance)[0], tf.shape(covariance)[-1]

  omega_offset = 1.0

  old_precision = tf.linalg.inv(old_covariance)
  target_precision = tf.linalg.inv(covariance)

  old_logdet = tf.math.log(tf.linalg.det(old_covariance) + 1e-15)
  kl_const_part = old_logdet - tf.cast(dim, dtype=covariance.dtype)


  ### LBFGS OPTIMIZATION ######################################################
  @tf.function
  def dual(eta):
    """Implements dual of optimization problem (Otto, 2021 Eq. 17) w.r.t. eta

    Args:
        eta ([tf.Tensor]): lagrangian multiplier eta [1]

    Returns:
        [(float, tf.Tensor)]: dual function and gradient 
          (Eq. 18 onwards, with simplification)
    """
    clipped_eta = tf.clip_by_value(tf.identity(eta), clip_value_min=0.0, clip_value_max=eta)
    # if eta < 0.0 and force_projection <= 0.0:
    #   eta = tf.constant([0.0], dtype=covariance.dtype)

    eta_squeezed, eta_expanded = tf.squeeze(clipped_eta), clipped_eta[:, tf.newaxis]

    # compute dual function
    new_precision = (eta_expanded * old_precision + target_precision) / (eta_expanded + omega_offset)
    new_covariance = tf.linalg.inv(new_precision)
    new_logdet = tf.math.log(tf.linalg.det(new_covariance))
    
    dual = eta_squeezed * eps - 0.5 * eta_squeezed * old_logdet + 0.5 * (eta_squeezed + omega_offset) * new_logdet
    
    # compute dual gradient
    trace_term = tf.linalg.trace(tf.linalg.matmul(old_precision, new_covariance))

    kl = 0.5 * (kl_const_part - new_logdet + trace_term)
    gradient = (eps - kl)

    return tf.reshape(dual, (batch_dim,)), tf.expand_dims(gradient, -1)

  # using vectorizable tfp::lbfgs optimizer for parallelization
  start = tf.zeros(shape=(batch_dim, 1), dtype=covariance.dtype)
  optim_results = tfp.optimizer.lbfgs_minimize(
    dual,
    initial_position=start,
    num_correction_pairs=20,
    max_iterations=500,
    parallel_iterations=64,
    max_line_search_iterations=50,
    tolerance=1e-5
  )

  eta_opt = tf.where(tf.math.is_nan(optim_results.position), tf.constant(0.0, dtype=tf.keras.backend.floatx()), optim_results.position)
  eta_opt = tf.clip_by_value(eta_opt, 0.0, 1e12)
  # eta_opt = tf.cond(force_projection <= 0.0, 
  #                   lambda : tf.clip_by_value(eta_opt, 0.0, 1e12),
  #                   lambda : tf.clip_by_value(eta_opt, -1e12, 1e12))

  # projection with optimal eta
  eta_opt_expanded = eta_opt[:, tf.newaxis]
  projected_precision = (eta_opt_expanded * old_precision + target_precision) / (eta_opt_expanded + omega_offset)
  projected_covariance = tf.linalg.inv(projected_precision)


  ### GRADIENT OF PROJECTION ##################################################
  def _custom_gradient(upstream):
    # compute non-zero eta KKT (deta_dQtarget), set zero projections to zero directly
    idx = tf.where(tf.greater(eta_opt, 0.0))[:, 0] # non-zero etas
    old_precision_, target_precision_, projected_covariance_, eta_opt_expanded_ = [
        tf.gather(params, idx)
        for params in [old_precision, target_precision, projected_covariance, eta_opt_expanded]
    ]

    dQ_deta_ = (omega_offset * old_precision_ - target_precision_) / (eta_opt_expanded_ + omega_offset)
    f2_dQ_ = tf.linalg.matmul(projected_covariance_, (tf.eye(dim, dtype=target_precision.dtype) - tf.linalg.matmul(old_precision_, projected_covariance_)))
    c_ = - 1.0 / tf.linalg.trace(tf.linalg.matmul(f2_dQ_, dQ_deta_)) # todo here
    deta_dQ_target_ = c_[:, tf.newaxis, tf.newaxis] * f2_dQ_
    
    deta_dQ_target = tf.tensor_scatter_nd_update(
      tf.zeros_like(projected_covariance), tf.expand_dims(idx, axis=-1),
      updates=deta_dQ_target_
    )
    
    # compute the rest gradients for the full input batch
    eo_squared = tf.square(omega_offset + eta_opt_expanded)
    dQ_deta = (omega_offset * old_precision - target_precision) / eo_squared

    d_Q = - tf.linalg.matmul(tf.linalg.matmul(projected_covariance, upstream), projected_covariance)
    d_eta = tf.linalg.trace(tf.linalg.matmul(d_Q, dQ_deta))

    d_Q_target = d_eta[:, tf.newaxis, tf.newaxis] * deta_dQ_target + d_Q / (omega_offset + eta_opt_expanded)
    d_cov_target = - tf.linalg.matmul(tf.linalg.matmul(target_precision, d_Q_target), target_precision)

    # note: only need the cov gradient, the rest doesn't matter
    return [d_cov_target, tf.zeros_like(old_covariance), tf.zeros_like(eps), tf.zeros_like(force_projection)]
  
  return projected_covariance, _custom_gradient
