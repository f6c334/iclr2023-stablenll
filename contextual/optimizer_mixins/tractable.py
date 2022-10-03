import tensorflow as tf

from common import utils
from optimizer_mixins.base_model import BaseModel


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
    lambda mean, covariance, other_mean, other_covariance: utils.mahalanobis_distance(
      other_mean, mean, tf.eye(tf.shape(mean)[-1], dtype=covariance.dtype)),
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
      covariance, other_covariance, tf.eye(tf.shape(covariance)[-1], dtype=covariance.dtype)),
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
class TractableOptimizerModel(BaseModel):

  def __init__(self,
               beta=1e-3,
               approximate_expm=True,
               mean_metric='mse',
               covariance_metric='mse',
               batch_reduce=tf.reduce_mean,
               **kwargs):
    super(TractableOptimizerModel, self).__init__(**kwargs)

    self.beta = beta
    self.approximate_expm = approximate_expm
    self.mean_loss = _MEAN_METRICS[mean_metric]
    self.covariance_loss = _COVARIANCE_METRICS[covariance_metric]
    self.batch_reduce = batch_reduce

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

    # regression onto the natural mean, covariance
    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)

      mean_loss = self.mean_loss(mean, covariance, natural_mean, natural_covariance)
      covariance_loss = self.covariance_loss(mean, covariance, natural_mean, natural_covariance)

      loss = self.batch_reduce(tf.reduce_sum(mean_loss + covariance_loss, axis=-1))

    # compute model gradients and apply
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x, y, (mean, covariance, A))
