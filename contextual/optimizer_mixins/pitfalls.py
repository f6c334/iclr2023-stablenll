import tensorflow as tf
import tensorflow_probability as tfp

from common import utils
from optimizer_mixins.base_model import BaseModel


###############################################################################
### OPT ALGOS #################################################################
class PitfallsOptimizerModel(BaseModel):

  def __init__(self, beta=0.5, **kwargs):
    super(PitfallsOptimizerModel, self).__init__(**kwargs)

    self.beta = beta

  def train_step(self, data):
    x, y = data

    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)

      nll_loss = self.pitfalls_nll(y, mean, covariance, self.beta)

    # compute model gradients and apply
    gradients = tape.gradient(nll_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x, y, (mean, covariance, A))

  @staticmethod
  @tf.custom_gradient
  def pitfalls_nll(x, mean, covariance, beta):
    loss = utils.gaussian_nll(x, mean, covariance, tf.reduce_mean)

    def _custom_gradient(upstream):
      batch_size = tf.cast(tf.shape(mean)[0], dtype=mean.dtype)

      # compute weighted covariance
      weighted_log_covariance = tf.cast(beta, dtype=tf.complex128) * tf.linalg.logm(tf.cast(covariance, tf.complex128))
      weighted_covariance = tf.cast(tf.linalg.expm(weighted_log_covariance), dtype=covariance.dtype)

      # compute nll gradients
      covariance_inv = tf.linalg.pinv(covariance)
      x_diff = tf.expand_dims(x - mean, axis=-1)
      x_mus = tf.linalg.matmul(x_diff, x_diff, transpose_b=True)  # elem-wise outer product

      dmean = -tf.linalg.matvec(covariance_inv, x - mean)
      dcovariance = -0.5 * tf.linalg.matmul(covariance_inv, tf.linalg.matmul(x_mus - covariance, covariance_inv))

      # compute weighted nll gradients
      weighted_dmean = tf.linalg.matvec(weighted_covariance, dmean)

      weighted_covariance_sqrt = tf.linalg.sqrtm(weighted_covariance)
      #weighted_dcovariance = tf.matmul(weighted_covariance, dcovariance)
      weighted_dcovariance = tf.matmul(weighted_covariance_sqrt, tf.matmul(dcovariance, weighted_covariance_sqrt))

      # ignoring gradients of x and beta
      # # gaussian_nll computes mean reduced loss, therefore mean reduced gradients
      # return [
      #   tf.zeros_like(x), (1.0 / batch_size) * upstream * weighted_dmean,
      #   (1.0 / batch_size) * upstream * weighted_dcovariance,
      #   tf.zeros_like(beta)
      # ]
      
      return tf.zeros_like(x), upstream * weighted_dmean, upstream * weighted_dcovariance, tf.zeros_like(beta)

    return loss, _custom_gradient