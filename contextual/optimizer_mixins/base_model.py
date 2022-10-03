import numpy as np
import tensorflow as tf

from common import utils


###############################################################################
### BASE MODEL ################################################################
class BaseModel(tf.keras.Model):

  def __init__(self, target_denormalizer=None, **kwargs):
    super(BaseModel, self).__init__(**kwargs)

    self.target_denormalizer = target_denormalizer

  @staticmethod
  def min_max_avg_dict(name, x):
    return {
      name + '/avg' : tf.reduce_mean(x),
      name + '/min' : tf.reduce_min(x),
      name + '/max' : tf.reduce_max(x),
    }
    
  def compute_metrics(self, x, y, y_pred, sample_weight=None):
    mean, covariance, A = y_pred
    
    # denormalize for metrics
    if self.target_denormalizer is not None:
      y_ = self.target_denormalizer(X=y)
      mean_, covariance_ = self.target_denormalizer(X=(mean, covariance))
    else:
      y_, mean_, covariance_ = y, mean, covariance
    
    un_mean_norm, mean_norm = tf.norm(mean_, axis=-1), tf.norm(mean, axis=-1)
    # metric_results |= self.min_max_avg_dict('norms/eucl_normalized_mean', mean_norm)

    un_covariance_norm, covariance_norm = tf.norm(covariance_, axis=[-2, -1]), tf.norm(covariance, axis=[-2, -1])
    # metric_results |= self.min_max_avg_dict('norms/frob_normalized_covariance', covariance_norm)
    
    return {
      'losses/unnormalized_nll' : utils.gaussian_nll(y_, mean_, covariance_),
      'losses/normalized_nll' : utils.gaussian_nll(y, mean, covariance),
      'losses/unnormalized_mse_mean' : tf.reduce_mean(tf.square(y_ - mean_)),
      'losses/normalized_mse_mean' : tf.reduce_mean(tf.square(y - mean)),
    } | self.min_max_avg_dict('norms/eucl_unnormalized_mean', un_mean_norm) \
      | self.min_max_avg_dict('norms/eucl_unnormalized_covariance', un_covariance_norm)