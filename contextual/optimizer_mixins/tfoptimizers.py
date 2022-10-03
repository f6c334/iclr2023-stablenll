import tensorflow as tf

from common import utils
from optimizer_mixins.base_model import BaseModel


###############################################################################
### OPT ALGOS #################################################################
class TFOptimizersModel(BaseModel):

  def __init__(self, **kwargs):
    super(TFOptimizersModel, self).__init__(**kwargs)

  def train_step(self, data):
    x, y = data

    with tf.GradientTape() as tape:
      mean, covariance, A = self(x, training=True)

      nll_loss = utils.gaussian_nll(y, mean, covariance)

    gradients = tape.gradient(nll_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return self.compute_metrics(x, y, (mean, covariance, A))