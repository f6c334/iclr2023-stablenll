import numpy as np
import tensorflow as tf

from common import utils
from common import callbacks as cb
from optimizer_mixins.base_model import BaseModel

_MEAN_METRICS = {
  'mse':
    lambda mean, covariance, other_mean, other_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.eye(tf.shape(mean)[-1])),
  'maha_pred':
    lambda mean, covariance, other_mean, other_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.linalg.inv(covariance)),
  'maha_target':
    lambda mean, covariance, other_mean, other_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.linalg.inv(other_covariance)),
}

_COVARIANCE_METRICS = {
  'mse':
    lambda mean, covariance, other_mean, other_covariance: tf.reduce_mean(tf.square(other_covariance - covariance),
                                                                          axis=(-1, -2)),
  'frob':
    lambda mean, covariance, other_mean, other_covariance: utils.frobenius_distance(other_covariance, covariance),
  'w2':
    lambda mean, covariance, other_mean, other_covariance: utils.wasserstein_distance(
      covariance, other_covariance, tf.eye(tf.shape(covariance)[-1])),
  'w2_pred':
    lambda mean, covariance, other_mean, other_covariance: utils.wasserstein_distance(
      covariance, other_covariance, tf.linalg.inv(covariance)),
  'kl_fw':  # KL(pred_cov, target_cov) [use with maha_target]
    lambda mean, covariance, other_mean, other_covariance: utils.kl_distance(covariance, other_covariance),
  'kl_bw':  # KL(target_cov, pred_cov) [use with maha_pred]
    lambda mean, covariance, other_mean, other_covariance: utils.kl_distance(other_covariance, covariance)
}


###############################################################################
### OPT ALGOS #################################################################
class TrustRegionsOptimizerModel(BaseModel):

  def __init__(self, proj_layer=None, advanced_metrics=True, **kwargs):
    super(TrustRegionsOptimizerModel, self).__init__(**kwargs)

    self.proj_layer = proj_layer
    self.advanced_metrics = advanced_metrics

  @tf.function
  def compute_metrics(self, x, y, y_pred, y_pred_proj, y_pred_old, sample_weight=None):
    metric_results = super().compute_metrics(x, y, y_pred=y_pred_proj, sample_weight=sample_weight)

    mean, covariance, _ = y_pred
    proj_mean, proj_covariance, _ = y_pred_proj
    old_mean, old_covariance = y_pred_old

    if self.advanced_metrics:
      identity = tf.eye(tf.shape(covariance)[-1], dtype=covariance.dtype)
      
      # metrics only make sense for normalized outputs ...
      return metric_results | {
        'trustregion_metrics/maha_mean': tf.reduce_mean(utils.mahalanobis_distance(old_mean, proj_mean, tf.linalg.inv(old_covariance))),
        'trustregion_metrics/frob_distance': tf.reduce_mean(utils.frobenius_distance(old_covariance, proj_covariance)),
        'trustregion_metrics/w2_distance': tf.reduce_mean(utils.wasserstein_distance(proj_covariance, old_covariance, identity)),
        'trustregion_metrics/w2old_distance': tf.reduce_mean(utils.wasserstein_distance(proj_covariance, old_covariance, tf.linalg.inv(old_covariance))),
        'trustregion_metrics/w2comm_distance': tf.reduce_mean(utils.wasserstein_distance_commutative(proj_covariance, old_covariance, tf.linalg.inv(old_covariance))),
        'trustregion_metrics/klfw_divergence': tf.reduce_mean(utils.kl_distance(proj_covariance, old_covariance)),
        'trustregion_metrics/klbw_divergence': tf.reduce_mean(utils.kl_distance(old_covariance, proj_covariance)),
        'trustregion_metrics/num_projections_mean': tf.math.count_nonzero(tf.reduce_sum(tf.square(proj_mean - mean), axis=-1)),
        'trustregion_metrics/num_projections_covariance': tf.math.count_nonzero(tf.reduce_sum(tf.square(proj_covariance - covariance), axis=[-2, -1]))
      }
    else:
      return metric_results


class TrueTrustRegionsOptimizerModel(TrustRegionsOptimizerModel):

  def __init__(self, **kwargs):
    super(TrueTrustRegionsOptimizerModel, self).__init__(**kwargs)

  @tf.function
  def train_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_means, idx)
    old_covariance = tf.gather(self.old_covariances, idx)

    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)
      proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

      loss = utils.gaussian_nll(y, proj_mean, proj_covariance)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x, y, y_pred=(mean, covariance, A), y_pred_proj=(proj_mean, proj_covariance, None), y_pred_old=(old_mean, old_covariance))

  def fit(self, x, y, callbacks=[], validation_data=None, **kwargs):
    save_predictions_callback = cb.SavePredictionsCallback(training_data=[x, y],
                                                           validation_data=validation_data,
                                                           proj_layer=self.proj_layer)

    idx = np.asarray(range(len(x)))

    init_means, init_covariances, init_A = self(x, training=False)
    self.old_means = tf.Variable(initial_value=init_means, trainable=False)
    self.old_covariances = tf.Variable(initial_value=init_covariances, trainable=False)

    if validation_data is not None:
      x_val, y_val = validation_data
      idx_val = np.asarray(range(len(x_val)))
      validation_data = ([idx_val, x_val], y_val)

      init_val_means, init_val_covariances, init_val_A = self(x_val, training=False)

      self.old_val_means = tf.Variable(initial_value=init_val_means, trainable=False)
      self.old_val_covariances = tf.Variable(initial_value=init_val_covariances, trainable=False)

    return super().fit(x=[idx, x],
                       y=y,
                       callbacks=callbacks + [save_predictions_callback],
                       validation_data=validation_data,
                       **kwargs)

  def evaluate(self, x, y, **kwargs):
    idx = np.asarray(range(len(x)))
    return super().evaluate(x=[idx, x], y=y, **kwargs)

  @tf.function
  def test_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_val_means, idx)
    old_covariance = tf.gather(self.old_val_covariances, idx)

    mean, covariance, A = self(x, training=False)
    proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

    return self.compute_metrics(x, y, y_pred=(mean, covariance, A), y_pred_proj=(proj_mean, proj_covariance, None), y_pred_old=(old_mean, old_covariance))


class AuxiliaryLossTrustRegionsOptimizerModel(TrustRegionsOptimizerModel):

  def __init__(self,
               proj_loss_coefficient=10.0,
               aux_mean_metric='maha_pred',
               aux_covariance_metric='kl_bw',
               batch_reduce=tf.reduce_mean,
               **kwargs):
    super(AuxiliaryLossTrustRegionsOptimizerModel, self).__init__(**kwargs)

    self.proj_loss_coefficient = proj_loss_coefficient

    self.mean_loss = _MEAN_METRICS[aux_mean_metric]
    self.covariance_loss = _COVARIANCE_METRICS[aux_covariance_metric]
    self.batch_reduce = batch_reduce

    self.old_theta = None

  def train_step(self, data):
    x, y = data

    # get old means, covariances for the trust region
    old_mean, old_covariance = self.compute_old(x)

    # compute gradient with trust region and auxiliary loss
    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)
      proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

      nll_loss = utils.gaussian_nll(y, proj_mean, proj_covariance)
      mean_loss = self.mean_loss(mean, covariance, tf.stop_gradient(proj_mean), tf.stop_gradient(proj_covariance))
      covariance_loss = self.covariance_loss(mean, covariance, tf.stop_gradient(proj_mean),
                                             tf.stop_gradient(proj_covariance))

      loss = nll_loss + self.proj_loss_coefficient * self.batch_reduce(
        tf.reduce_sum(mean_loss + covariance_loss, axis=-1))

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # NOTE: A is not necessarily the same as chol(Sigma_projected)
    return self.compute_metrics(x, y, y_pred=(mean, covariance, A), y_pred_old=(old_mean, old_covariance, A))

  def fit(self, x, y, callbacks=[], **kwargs):
    _ = self(x[0:1], training=False)  # build weights if needed
    self.old_theta = [tf.Variable(initial_value=value, trainable=False) for value in self.trainable_variables]

    save_theta_callback = tf.keras.callbacks.LambdaCallback(
      on_epoch_begin=lambda epoch, logs: utils.nested_assign(self.old_theta, self.trainable_variables),
    )

    return super().fit(x=x, y=y, callbacks=callbacks + [save_theta_callback], **kwargs)

  def set_trainable_variables(self, new_trainable_variables: list):
    for variable, new_variable in zip(self.trainable_variables, new_trainable_variables):
      variable.assign(new_variable)

  def test_step(self, data):
    x, y = data

    old_mean, old_covariance = self.compute_old(x)

    mean, covariance, A = self(x, training=False)

    return self.compute_metrics(x, y, y_pred=(mean, covariance, A), y_pred_old=(old_mean, old_covariance, A))

  def compute_old(self, x):
    self.theta = tf.identity_n(self.trainable_variables)

    # get old means, covariances for the trust region
    self.set_trainable_variables(self.old_theta)
    old_mean, old_covariance, _ = self(x, training=False)

    # set weights back to current theta
    self.set_trainable_variables(self.theta)

    return old_mean, old_covariance


class NBackTrustRegionsOptimizerModel(TrustRegionsOptimizerModel):

  def __init__(self, proj_layer=None, n_back=5, **kwargs):
    super(NBackTrustRegionsOptimizerModel, self).__init__(**kwargs)

    self.proj_layer = proj_layer
    self.n_back = n_back

  def train_step(self, data):
    x, y = data

    self.theta = tf.identity_n(self.trainable_variables)

    # get old means, covariances for the trust region
    old_mean, old_covariance = self.compute_old(x)

    # compute loss and gradient with new predictions plus trust region
    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)
      proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

      loss = utils.gaussian_nll(y, proj_mean, proj_covariance)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # NOTE: A is not necessarily the same as chol(Sigma_projected)
    return self.compute_metrics(x, y, y_pred=(proj_mean, proj_covariance, A), y_pred_old=(old_mean, old_covariance, A))

  def test_step(self, data):
    x, y = data

    self.theta = tf.identity_n(self.trainable_variables)

    # get old means, covariances for the trust region
    old_mean, old_covariance = self.compute_old(x)

    # compute evaluation parameters
    mean, covariance, A = self(x, training=False)
    proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

    return self.compute_metrics(x, y, y_pred=(proj_mean, proj_covariance, A), y_pred_old=(old_mean, old_covariance, A))

  def fit(self, x, y, callbacks=[], **kwargs):
    _ = self(x[0:1], training=False)  # build weights if needed
    self.old_thetas = [[tf.Variable(initial_value=value, trainable=False)
                        for value in self.trainable_variables]
                       for _ in range(self.n_back)]

    def circular_assign(epoch, logs):
      for i in range(self.n_back - 1):
        utils.nested_assign(self.old_thetas[i], self.old_thetas[i + 1])
      utils.nested_assign(self.old_thetas[-1], self.trainable_variables)

    save_theta_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=circular_assign,)

    return super().fit(x=x, y=y, callbacks=callbacks + [save_theta_callback], **kwargs)

  def compute_old(self, x):
    self.theta = tf.identity_n(self.trainable_variables)

    # get old means, covariances for the trust region
    self.set_trainable_variables(self.old_thetas[0])
    old_mean, old_covariance, _ = self(x, training=False)

    for i in range(1, self.n_back):
      self.set_trainable_variables(self.old_thetas[i])
      mean, covariance, _ = self(x, training=False)
      old_mean, old_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

    # set theta to current value
    self.set_trainable_variables(self.theta)

    return old_mean, old_covariance

  def set_trainable_variables(self, new_trainable_variables: list):
    for variable, new_variable in zip(self.trainable_variables, new_trainable_variables):
      variable.assign(new_variable)


class ComparisonNbackTrueTrustRegionsOptimizerModel(TrueTrustRegionsOptimizerModel):

  def __init__(self, n_back=5, **kwargs):
    super(ComparisonNbackTrueTrustRegionsOptimizerModel, self).__init__(**kwargs)

    self.n_back = n_back

  def train_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_means, idx)
    old_covariance = tf.gather(self.old_covariances, idx)

    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)
      proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

      loss = utils.gaussian_nll(y, proj_mean, proj_covariance)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # NOTE: A is not necessarily the same as chol(Sigma_projected)
    return self.compute_metrics(x,
                                y,
                                y_pred=(proj_mean, proj_covariance, A),
                                y_pred_old=(old_mean, old_covariance, A),
                                old_projections=self.compute_old(x))

  def test_step(self, data):
    (idx, x), y = data

    old_mean = tf.gather(self.old_val_means, idx)
    old_covariance = tf.gather(self.old_val_covariances, idx)

    mean, covariance, A = self(x, training=False)
    proj_mean, proj_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)

    return self.compute_metrics(x,
                                y,
                                y_pred=(proj_mean, proj_covariance, A),
                                y_pred_old=(old_mean, old_covariance, A),
                                old_projections=self.compute_old(x))

  def compute_metrics(self, x, y, y_pred, y_pred_old, old_projections, sample_weight=None):
    metric_results = super().compute_metrics(x, y, y_pred, y_pred_old, sample_weight=sample_weight)

    mean, covariance, _ = y_pred
    old_mean, old_covariance, _ = y_pred_old  # true old mean / old covariance

    # denormalize for metrics
    if self.target_denormalizer is not None:
      mean_, covariance_ = self.target_denormalizer(X=(mean, covariance))
      old_mean_, old_covariance_ = self.target_denormalizer(X=(old_mean, old_covariance))
      old_projections_ = [self.target_denormalizer(X=(m, c)) for m, c in old_projections]
    else:
      y_, mean_, covariance_ = y, mean, covariance
      old_mean_, old_covariance_ = old_mean, old_covariance
      old_projections_ = old_projections

    klfw_nback = [
      0.5 * tf.reduce_mean((utils.mahalanobis_distance(old_mean_, m, tf.linalg.inv(old_covariance_)) +
                            utils.kl_distance(c, old_covariance_))) for m, c in old_projections_
    ]

    klbw_nback = [
      0.5 * tf.reduce_mean(
        (utils.mahalanobis_distance(m, old_mean_, tf.linalg.inv(c)) + utils.kl_distance(c, old_covariance_)))
      for m, c in old_projections_
    ]

    klfw_nback_dict = {f'klfw_nback_{i}': kl for i, kl in enumerate(klfw_nback)}
    klbw_nback_dict = {f'klbw_nback_{i}': kl for i, kl in enumerate(klbw_nback)}

    return metric_results | klfw_nback_dict | klbw_nback_dict

  def fit(self, x, y, callbacks=[], **kwargs):
    _ = self(x[0:1], training=False)  # build weights if needed
    self.old_thetas = [[tf.Variable(initial_value=value, trainable=False)
                        for value in self.trainable_variables]
                       for _ in range(self.n_back)]

    def circular_assign(epoch, logs):
      for i in range(self.n_back - 1):
        utils.nested_assign(self.old_thetas[i], self.old_thetas[i + 1])
      utils.nested_assign(self.old_thetas[-1], self.trainable_variables)

    save_theta_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=circular_assign,)

    return super().fit(x=x, y=y, callbacks=callbacks + [save_theta_callback], **kwargs)

  def compute_old(self, x):
    self.theta = tf.identity_n(self.trainable_variables)

    # get old predictions for each network in the memory
    old_predictions = []
    for i in range(self.n_back):
      self.set_trainable_variables(self.old_thetas[i])
      mean, covariance, _ = self(x, training=False)
      old_predictions.append([mean, covariance])

    # set theta to current value
    self.set_trainable_variables(self.theta)

    # get projected predictions for each network
    old_projections = []
    for i in range(self.n_back):
      old_mean, old_covariance = old_predictions[i]
      for j in range(i + 1, self.n_back):
        mean, covariance = old_predictions[j]
        old_mean, old_covariance = self.proj_layer(mean, old_mean, covariance, old_covariance)
      old_projections.append([old_mean, old_covariance])

    return old_projections

  def set_trainable_variables(self, new_trainable_variables: list):
    for variable, new_variable in zip(self.trainable_variables, new_trainable_variables):
      variable.assign(new_variable)