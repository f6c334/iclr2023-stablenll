import tensorflow as tf
import tensorflow_probability as tfp

from common import utils
from optimizer_mixins.base_model import BaseModel


###############################################################################
### OPT ALGOS #################################################################
# TODO: properly implement, should generally work but
# - computation of metrics not correct
# - old parameters should be computed using a saved version of theta (via callback)
# - therefore adjust the fit function aswell
class ConjugateGradientOptimizerModel(BaseModel):

  def __init__(self, tr_value=1e-3, cg_damping=1e-5, cg_max_iterations=20, cg_residual_tolerance=1e-5, ls_max_iterations=20, ls_coefficient=0.6, **kwargs):
    super(ConjugateGradientOptimizerModel, self).__init__(**kwargs)

    self.tr_value = tr_value

    self.cg_damping = cg_damping
    self.cg_max_iterations = cg_max_iterations
    self.cg_residual_tolerance = cg_residual_tolerance

    self.ls_max_iterations = ls_max_iterations
    self.ls_coefficient = ls_coefficient

    self.cg_residual_tracker = tf.keras.metrics.Mean(name='cg_residual')
    self.kl_div_tracker = tf.keras.metrics.Mean(name='kl_div')

  def train_step(self, data):
    x, y = data

    # Save old predictions and parameters
    old_mean, _, old_covariance_cholesky = self(x, training=False)
    old_theta = tf.identity_n(self.trainable_variables)

    # define loss, kl and hessian vector product function
    def nll_loss_fn(theta):
      self.set_trainable_variables(theta)
      mean, _, covariance_cholesky = self(x, training=True)
      return utils.gauss_nll(y, mean, covariance_cholesky)

    def kl_fn(theta):
      self.set_trainable_variables(theta)
      mean, _, covariance_cholesky = self(x, training=True)
      return utils.kl_divergence(mean, covariance_cholesky, old_mean, old_covariance_cholesky)
    
    def kl_hessian_vector_product(vector):
      with tf.GradientTape() as tape:
        with tf.GradientTape() as tape_:
          loss = kl_fn(self.trainable_variables)
        gradients = tape_.gradient(loss, self.trainable_variables)
        flattened_gradient = utils.nested_flatten(gradients)
        
        gradient_vector_product = tf.reduce_sum(flattened_gradient * vector)
      hessian_vector_product = tape.gradient(gradient_vector_product, self.trainable_variables)

      return utils.nested_flatten(hessian_vector_product) + self.cg_damping * vector

    # Compute original gradient for nll loss, w.r.t. old parameters
    with tf.GradientTape() as tape:
      nll_loss = nll_loss_fn(old_theta)
    
    gradients = tape.gradient(nll_loss, self.trainable_variables)
    flat_gradients = utils.nested_flatten(gradients)
    
    # Compute conjugate gradient and its optimal step size (see TRPO)
    flat_conj_grad, residual_error = utils.conjugate_gradient(
      Ax=kl_hessian_vector_product,
      b=flat_gradients,
      x0=tf.zeros_like(flat_gradients),
      max_iterations=self.cg_max_iterations,
      residual_tolerance=self.cg_residual_tolerance
    )

    alpha = tf.sqrt((2 * self.tr_value) / tf.tensordot(flat_gradients, flat_conj_grad, axes=1))
    full_step = - alpha * flat_conj_grad

    # Compute true gradient by line search along conjugate
    flat_new_theta, kl_div = utils.linesearch(
      loss_fn=lambda flat_theta : nll_loss_fn(utils.nested_unflatten(old_theta, flat_theta)),
      constraint_fn=lambda flat_theta : kl_fn(utils.nested_unflatten(old_theta, flat_theta)),
      x=utils.nested_flatten(old_theta),
      full_step=full_step,
      delta=self.tr_value,
      max_iterations=self.ls_max_iterations,
      backtrack_coefficient=self.ls_coefficient
    )
    
    # Set new network parameters
    self.set_trainable_variables(utils.nested_unflatten(old_theta, flat_new_theta))
    
    # Compute own metrics
    old_covariance = tf.linalg.matmul(old_covariance_cholesky, old_covariance_cholesky, transpose_b=True)
    return self.compute_metrics(x, y, (old_mean, old_covariance, None))
    # return self.compute_metrics(x, y, (mean, covariance, A), residual_error, kl_div)
    # return self.compute_metrics(nll_loss, (y_true_mean, old_mean), (y_true_covariance, old_covariance), residual_error, kl_div)
  
  def set_trainable_variables(self, new_trainable_variables : list):
    for variable, new_variable in zip(self.trainable_variables, new_trainable_variables):
      variable.assign(new_variable)
  
  # def compute_metrics(self, nll_loss, mean_true_pred, covariance_true_pred, cg_residual, kl_div):
  #   return super().compute_metrics(nll_loss, mean_true_pred, covariance_true_pred) | {'cg_residual' : cg_residual, 'kl_div' : kl_div}