import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from common import utils
from optimizer_mixins.base_model import BaseModel
from optimizer_mixins.trustregions import TrueTrustRegionsOptimizerModel


def approx_expm(X):
  return tf.eye(X.shape[-1], dtype=X.dtype) + X + tf.linalg.matmul(X, X) / 2


def _tractable(mu_t, A_t, delta, M, approximate_expm: bool = False):
  mu = mu_t + tf.linalg.matvec(A_t, delta)

  if approximate_expm:
    exp_mapping = approx_expm(0.5 * M)
  else:
    exp_mapping = tf.linalg.expm(0.5 * M)

  A = tf.linalg.matmul(A_t, exp_mapping)
  return mu, tf.linalg.matmul(A, A, transpose_b=True), A


_MEAN_METRICS = {
  'mse':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.eye(tf.shape(mean)[-1], dtype=covariance.dtype)),
  'maha_pred':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.linalg.inv(covariance)),
  'maha_target':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.linalg.inv(other_covariance)),
  'maha_old':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.linalg.inv(old_covariance)),
}

_COVARIANCE_METRICS = {
  'mse':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: tf.reduce_mean(
      tf.square(other_covariance - covariance), axis=(-1, -2)),
  'frob':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.frobenius_distance(
      other_covariance, covariance),
  'w2':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.wasserstein_distance(
      covariance, other_covariance, tf.eye(tf.shape(covariance)[-1], dtype=covariance.dtype)),
  'w2_comm':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.wasserstein_distance_commutative(
      covariance, other_covariance, tf.eye(tf.shape(covariance)[-1], dtype=covariance.dtype)),
  'w2_pred':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.wasserstein_distance(
      covariance, other_covariance, tf.linalg.inv(covariance)),
  'w2_old':
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.wasserstein_distance(
      covariance, other_covariance, tf.linalg.inv(old_covariance)),
  'kl_fw':  # KL(pred_cov, target_cov) [use with maha_target]
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.kl_distance(
      covariance, other_covariance),
  'kl_bw':  # KL(target_cov, pred_cov) [use with maha_pred]
    lambda mean, covariance, other_mean, other_covariance, old_mean, old_covariance: utils.kl_distance(
      other_covariance, covariance)
}


###############################################################################
### OPT ALGOS #################################################################
class TraptableOptimizerModel(TrueTrustRegionsOptimizerModel):

  def __init__(self,
               beta=1e-3,
               local_covariance_gradient_scale=1.0,
               approximate_expm=True,
               project_natural_parameters=True,
               regress_on_projected_parameters=True,
               use_tractable_before_projection=True,
               mean_metric='mse',
               covariance_metric='mse',
               batch_reduce=tf.reduce_mean,
               **kwargs):
    super().__init__(**kwargs)

    self.beta = beta
    self.local_covariance_gradient_scale = local_covariance_gradient_scale
    self.approximate_expm = approximate_expm
    self.project_natural_parameters = project_natural_parameters
    self.regress_on_projected_parameters = regress_on_projected_parameters
    self.use_tractable_before_projection = use_tractable_before_projection
    self.mean_loss = _MEAN_METRICS[mean_metric]
    self.covariance_loss = _COVARIANCE_METRICS[covariance_metric]
    self.batch_reduce = batch_reduce

  @tf.function
  def train_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_means, idx)
    old_covariance = tf.gather(self.old_covariances, idx)

    mean_t, covariance_t, A_t = self(x, training=False)
    delta, M = tf.zeros_like(mean_t), tf.zeros_like(A_t)

    with tf.GradientTape() as tape:
      tape.watch([delta, M])

      if self.use_tractable_before_projection:
        mean, covariance, _ = _tractable(mean_t, A_t, delta, M, self.approximate_expm)
      else:
        mean, covariance = mean_t, covariance_t

      proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

      # projmean, proj_cov would be _t variants and be used in tractable
      if self.use_tractable_before_projection:
        mean_, covariance_ = proj_mean, proj_covariance
      else:
        mean_, covariance_, _ = _tractable(proj_mean, tf.linalg.cholesky(proj_covariance), delta, M,
                                           self.approximate_expm)

      nll_loss = utils.gaussian_nll(y, mean_, covariance_)

    [d_delta, d_M] = tape.gradient(nll_loss, [delta, M])

    if self.use_tractable_before_projection:
      natural_mean, natural_covariance, _ = _tractable(mean_t, A_t, -self.beta * d_delta, 
                                                       -self.beta * self.local_covariance_gradient_scale * d_M,
                                                       self.approximate_expm)
    else:
      natural_mean, natural_covariance, _ = _tractable(proj_mean, tf.linalg.cholesky(proj_covariance),
                                                       -self.beta * d_delta, 
                                                       -self.beta * self.local_covariance_gradient_scale * d_M, self.approximate_expm)


    if self.project_natural_parameters:
      next_mean, next_covariance = self.proj_layer(natural_mean, old_mean, natural_covariance, old_covariance)
      next_covariance = tf.where(tf.math.is_nan(tf.linalg.sqrtm(natural_covariance)), proj_covariance, next_covariance)
    else:
      next_mean, next_covariance = natural_mean, natural_covariance
    
    
    with tf.GradientTape() as tape:
      mean, covariance, _ = self(x, training=True)

      if self.regress_on_projected_parameters:
        current_mean, current_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)
      else:
        current_mean, current_covariance = mean, covariance

      mean_loss = self.mean_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean, old_covariance)
      covariance_loss = self.covariance_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean,
                                             old_covariance)
      
      loss = self.batch_reduce(tf.reduce_sum(mean_loss + covariance_loss, axis=-1))
    
    # tf.print('Nans/Inf Loss : ', tf.reduce_any(tf.math.is_nan(loss)), tf.reduce_any(tf.math.is_inf(loss)))
    # tf.print('Nans Mean Loss : ', tf.reduce_any(tf.math.is_nan(mean_loss)))
    # tf.print('Nans Covariance Loss : ', tf.reduce_any(tf.math.is_nan(covariance_loss)))
    # tf.print('Nans Projected Covariance : ', tf.reduce_any(tf.math.is_nan(proj_covariance)))
    # tf.print('Nans Covariance : ', tf.reduce_any(tf.math.is_nan(covariance)))
    
    # nan_idx = tf.where(tf.math.is_nan(covariance_loss))
    # if tf.size(nan_idx) > 0:
    #   tf.print(nan_idx)
    #   nan_id = nan_idx[0][0]
    #   tf.print(old_covariance[nan_id], current_covariance[nan_id], natural_covariance[nan_id], next_covariance[nan_id])
    
    # compute model gradients and apply
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x,
                                y,
                                y_pred=(mean_t, covariance_t, A_t),
                                y_pred_proj=(proj_mean, proj_covariance, None),
                                y_pred_old=(old_mean, old_covariance),
                                local_gradients=(d_delta, d_M),
                                y_natural=(natural_mean, natural_covariance),
                                y_targets=(next_mean, next_covariance))

  @tf.function
  def compute_metrics(self, x, y, y_pred, y_pred_proj, y_pred_old, local_gradients=None, y_natural=None, y_targets=None, sample_weight=None):
    metric_results = super().compute_metrics(x, y, y_pred, y_pred_proj, y_pred_old, sample_weight=sample_weight)

    mean, covariance, _ = y_pred
    proj_mean, proj_covariance, _ = y_pred_proj
    old_mean, old_covariance = y_pred_old

    if self.advanced_metrics and local_gradients is not None: # loc != None means also natural and targets are not None
      d_delta, d_M = local_gradients
      natural_mean, natural_covariance = y_natural
      next_mean, next_covariance = y_targets
      
      batch_size = tf.shape(covariance)[0]
      num_legal_projections = tf.math.count_nonzero(tf.reduce_sum(tf.square(proj_covariance - next_covariance), axis=[-2, -1]), dtype=batch_size.dtype)
      
      return metric_results | {
        'tractable_metrics/num_illegal_tractable_projections' : batch_size - num_legal_projections,
      } | self.min_max_avg_dict('tractable_metrics/eucl_d_delta', tf.norm(d_delta, axis=-1)) \
        | self.min_max_avg_dict('tractable_metrics/eucl_d_M', tf.norm(d_M, axis=[-2, -1])) \
        | self.min_max_avg_dict('tractable_metrics/eucl_natural_covariance', tf.norm(natural_covariance, axis=[-2, -1]))
    else:
      return metric_results


class TraptableVariantOptimizerModel(BaseModel):

  def __init__(self,
               proj_layer=None,
               beta=1e-3,
               approximate_expm=True,
               mean_metric='mse',
               covariance_metric='mse',
               batch_reduce=tf.reduce_mean,
               **kwargs):
    super().__init__(**kwargs)

    self.proj_layer = proj_layer
    self.beta = beta
    self.approximate_expm = approximate_expm
    self.mean_loss = _MEAN_METRICS[mean_metric]
    self.covariance_loss = _COVARIANCE_METRICS[covariance_metric]
    self.batch_reduce = batch_reduce

  @tf.function
  def train_step(self, data):
    x, y = data

    # compute mean, covariance and from that natural mean, covariance through tractable
    mean_t, _, A_t = self(x, training=False)
    delta, M = tf.zeros_like(mean_t), tf.zeros_like(A_t)

    with tf.GradientTape() as tape:
      tape.watch([delta, M])

      mean, covariance, _ = _tractable(mean_t, A_t, delta, M, self.approximate_expm)

      nll_loss = utils.gaussian_nll(y, mean, covariance)

    [d_delta, d_M] = tape.gradient(nll_loss, [delta, M])
    natural_mean, natural_covariance, _ = _tractable(mean_t, A_t, -self.beta * d_delta, -self.beta * d_M,
                                                     self.approximate_expm)
    next_mean, next_covariance = self.proj_layer(natural_mean, mean, natural_covariance, covariance)

    # regression onto the natural mean, covariance
    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)

      mean_loss = self.mean_loss(mean, covariance, next_mean, next_covariance, None, None)
      covariance_loss = self.covariance_loss(mean, covariance, next_mean, next_covariance, None, None)

      loss = self.batch_reduce(tf.reduce_sum(mean_loss + covariance_loss, axis=-1))

    # compute model gradients and apply
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x, y, (mean, covariance, A))



class MultipleTraptableOptimizerModel(TrueTrustRegionsOptimizerModel):

  def __init__(self,
               beta=1e-3,
               approximate_expm=True,
               regress_on_projected_parameters=True,
               tractable_updates=True,
               mean_metric='mse',
               covariance_metric='mse',
               batch_reduce=tf.reduce_mean,
               **kwargs):
    super().__init__(**kwargs)

    self.beta = beta
    self.approximate_expm = approximate_expm
    self.tractable_updates = tractable_updates
    self.regress_on_projected_parameters = regress_on_projected_parameters
    self.mean_loss = _MEAN_METRICS[mean_metric]
    self.covariance_loss = _COVARIANCE_METRICS[covariance_metric]
    self.batch_reduce = batch_reduce

  @tf.function
  def train_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_means, idx)
    old_covariance = tf.gather(self.old_covariances, idx)

    mean_t, covariance_t, A_t = self(x, training=False)
    delta, M = tf.zeros_like(mean_t), tf.zeros_like(A_t)

    natural_mean, natural_covariance, natural_A = mean_t, covariance_t, A_t
    for _ in range(self.tractable_updates):
      with tf.GradientTape() as tape:
        tape.watch([delta, M])

        mean, covariance, _ = _tractable(natural_mean, natural_A, delta, M, self.approximate_expm)
        proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

        nll_loss = utils.gaussian_nll(y, proj_mean, proj_covariance)

      [d_delta, d_M] = tape.gradient(nll_loss, [delta, M])

      natural_mean, natural_covariance, natural_A = _tractable(natural_mean, natural_A, -self.beta * d_delta, -self.beta * d_M,
                                                        self.approximate_expm)

    next_mean, next_covariance = self.proj_layer(natural_mean, old_mean, natural_covariance, old_covariance)

    with tf.GradientTape() as tape:
      mean, covariance, _ = self(x, training=True)

      if self.regress_on_projected_parameters:
        current_mean, current_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)
      else:
        current_mean, current_covariance = mean, covariance

      mean_loss = self.mean_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean, old_covariance)
      covariance_loss = self.covariance_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean,
                                             old_covariance)

      loss = self.batch_reduce(tf.reduce_sum(mean_loss + covariance_loss, axis=-1))

    # compute model gradients and apply
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x,
                                y,
                                y_pred=(proj_mean, proj_covariance, A_t),
                                y_pred_old=(old_mean, old_covariance, A_t))










































  # @tf.function
  # def train_step(self, data):
  #   (idx, x), y = data

  #   old_mean = tf.gather(self.old_means, idx)
  #   old_covariance = tf.gather(self.old_covariances, idx)

  #   mean_t, covariance_t, A_t = self(x, training=False)
  #   delta, M = tf.zeros_like(mean_t), tf.zeros_like(A_t)

  #   with tf.GradientTape() as tape:
  #     tape.watch([delta, M])

  #     if self.use_tractable_before_projection:
  #       mean, covariance, _ = _tractable(mean_t, A_t, delta, M, self.approximate_expm)
  #     else:
  #       mean, covariance = mean_t, covariance_t

  #     proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

  #     # projmean, proj_cov would be _t variants and be used in tractable
  #     if self.use_tractable_before_projection:
  #       mean_, covariance_ = proj_mean, proj_covariance
  #     else:
  #       mean_, covariance_, _ = _tractable(proj_mean, tf.linalg.cholesky(proj_covariance), delta, M,
  #                                          self.approximate_expm)

  #     tf.print('Nans/Infs NLL : ', tf.reduce_any(tf.math.is_nan(covariance_)), tf.reduce_any(tf.math.is_inf(covariance_)))
  #     nll_loss = utils.gaussian_nll(y, mean_, covariance_)

  #   [d_delta, d_M] = tape.gradient(nll_loss, [delta, M])

  #   if self.use_tractable_before_projection:
  #     natural_mean, natural_covariance, _ = _tractable(mean_t, A_t, -self.beta * d_delta, -self.beta * d_M,
  #                                                      self.approximate_expm)
  #   else:
  #     natural_mean, natural_covariance, _ = _tractable(proj_mean, tf.linalg.cholesky(proj_covariance),
  #                                                      -self.beta * d_delta, -self.beta * d_M, self.approximate_expm)


  #   if self.project_natural_parameters:
  #     next_mean, next_covariance = self.proj_layer(natural_mean, old_mean, natural_covariance, old_covariance)
  #     next_covariance = tf.where(tf.math.is_nan(tf.linalg.sqrtm(natural_covariance)), proj_covariance, next_covariance)
  #   else:
  #     next_mean, next_covariance = natural_mean, natural_covariance
      
          
  #   with tf.GradientTape() as tape:
  #     mean, covariance, _ = self(x, training=True)

  #     if self.regress_on_projected_parameters:
  #       current_mean, current_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)
  #     else:
  #       current_mean, current_covariance = mean, covariance

  #     mean_loss = self.mean_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean, old_covariance)
  #     covariance_loss = self.covariance_loss(current_mean, current_covariance, next_mean, next_covariance, old_mean,
  #                                            old_covariance)
  #     # mean_loss = tf.where(tf.math.is_nan(covariance_loss), tf.ones_like(mean_loss), mean_loss)
  #     # covariance_loss = tf.where(tf.math.is_nan(covariance_loss), tf.ones_like(covariance_loss), covariance_loss)
      
  #     loss = self.batch_reduce(tf.reduce_sum(mean_loss + covariance_loss, axis=-1))

  #   # compute model gradients and apply
  #   tf.print('Nans/Inf Loss : ', tf.reduce_any(tf.math.is_nan(loss)), tf.reduce_any(tf.math.is_inf(loss)))
  #   tf.print('Nans Mean Loss : ', tf.reduce_any(tf.math.is_nan(mean_loss)))
  #   tf.print('Nans Covariance Loss : ', tf.reduce_any(tf.math.is_nan(covariance_loss)))
  #   tf.print('Nans Projected Covariance : ', tf.reduce_any(tf.math.is_nan(proj_covariance)))
  #   tf.print('Nans Covariance : ', tf.reduce_any(tf.math.is_nan(covariance)))
    
  #   nan_idx = tf.where(tf.math.is_nan(covariance_loss))
  #   if tf.size(nan_idx) > 0:
  #     tf.print(nan_idx)
  #     nan_id = nan_idx[0][0]
  #     tf.print(old_covariance[nan_id], current_covariance[nan_id], natural_covariance[nan_id], next_covariance[nan_id])
    
  #   gradients = tape.gradient(loss, self.trainable_variables)
  #   tf.print('Gradient Nans : ', [tf.reduce_any(tf.math.is_nan(var)) for var in gradients])
  #   self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
  #   tf.print('Network Param Nans : ', [tf.reduce_any(tf.math.is_nan(var)) for var in self.trainable_variables])