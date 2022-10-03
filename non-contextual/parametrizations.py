import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple

"""
- HOW to define parametrizations ?
-- either per function or per class
- HOW to define optimization method ?





: class parametrization:
- then wed need call method to get mean, var which has as parameters the parametrization
- could save more things like last elems etc
"""


class GaussParametrization(object):

  def __init__(self) -> None:
    super().__init__()
  
  def __call__(self, p_mean, p_covariance, update: bool = False):
    raise NotImplementedError('Base class, use sub class')
  
  def inverse(self, mean, covariance):
    raise NotImplementedError('Base class, use sub class')

class VanillaParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, p_covariance
  
  def inverse(self, mean, covariance):
    return mean, covariance

class CholeskyParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.matmul(p_covariance, tf.transpose(p_covariance))
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.cholesky(covariance)

class DiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(p_covariance)
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.diag_part(covariance)

class LogDiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.math.exp(p_covariance))
  
  def inverse(self, mean, covariance):
    return mean, tf.math.log(tf.linalg.diag_part(covariance))

class NegLogDiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.math.exp(- p_covariance))
  
  def inverse(self, mean, covariance):
    return mean, - tf.math.log(tf.linalg.diag_part(covariance))

class InvLogDiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.math.exp(1.0 / p_covariance))
  
  def inverse(self, mean, covariance):
    return mean,  1.0 / tf.math.log(tf.linalg.diag_part(covariance))

class ExpDiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.math.log(p_covariance))
  
  def inverse(self, mean, covariance):
    return mean, tf.math.exp(tf.linalg.diag_part(covariance))

class SqrtCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.matmul(p_covariance, p_covariance)
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.sqrtm(covariance)

class DiagPrecisionParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(1.0 / p_covariance)
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.diag_part(1.0 / covariance)

class DiagNegPrecisionParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(- 1.0 / p_covariance)
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.diag_part(- 1.0 / covariance)

class LogSqrtDiagCovarianceParametrization(GaussParametrization): # i.e. logstd

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.square(tf.math.exp(p_covariance)))
  
  def inverse(self, mean, covariance):
    return mean, tf.math.log(tf.sqrt(tf.linalg.diag_part(covariance)))

class SoftplusDiagCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.diag(tf.math.softplus(p_covariance))
  
  def inverse(self, mean, covariance):
    return mean, tfp.math.softplus_inverse(tf.linalg.diag_part(covariance))

class TractableImitationParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    sqrt_covariance = tf.linalg.expm(p_covariance)
    covariance = tf.linalg.matmul(sqrt_covariance, sqrt_covariance)
    return tf.linalg.matvec(covariance, p_mean), covariance
  
  def inverse(self, mean, covariance):
    covariance_inv = tf.linalg.inv(covariance)
    sqrt_covariance = tf.linalg.sqrtm(covariance)
    log_covariance = tf.linalg.logm(tf.cast(sqrt_covariance, dtype=tf.complex64))
    log_covariance = tf.cast(log_covariance, dtype=covariance.dtype)
    return tf.linalg.matvec(covariance_inv, mean), log_covariance

class DiagTractableImitationParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return tf.exp(tf.math.log(p_mean) * tf.exp(p_covariance)), tf.linalg.diag(tf.exp(p_covariance))
  
  def inverse(self, mean, covariance):
    return - tf.math.log(mean) / tf.linalg.diag_part(covariance), tf.math.log(tf.linalg.diag_part(covariance))

class LogCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.expm(p_covariance)
  
  def inverse(self, mean, covariance):
    log_covariance = tf.linalg.logm(tf.cast(covariance, dtype=tf.complex64))
    return mean, tf.cast(log_covariance, dtype=covariance.dtype)

class PrecisionParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    return p_mean, tf.linalg.pinv(p_covariance)
  
  def inverse(self, mean, covariance):
    return mean, tf.linalg.pinv(covariance)

class LogSqrtCovarianceParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance):
    sqrt_covariance = tf.linalg.expm(p_covariance)
    return p_mean, tf.matmul(sqrt_covariance, sqrt_covariance)
  
  def inverse(self, mean, covariance):
    sqrt_covariance = tf.linalg.sqrtm(covariance)
    sqrt_covariance = tf.cast(sqrt_covariance, dtype=tf.complex64)
    return mean,  tf.cast(tf.linalg.logm(sqrt_covariance), dtype=tf.float64)

class TractableCholeskyParametrization(GaussParametrization):

  def __call__(self, p_mean, p_covariance, delta, M):
    mu = p_mean + tf.linalg.matvec(p_covariance, delta)
    A = tf.linalg.matmul(p_covariance, tf.linalg.expm(0.5 * M))
    
    return mu, tf.matmul(A, tf.transpose(A))

  def inverse(self, mean, covariance):
    return mean, tf.linalg.cholesky(covariance)

class ApproximateTractableCholeskyParametrization(GaussParametrization):

  def exp_map(X):
    return tf.eye(tf.shape(X)[-1], dtype=X.dtype) + X + tf.linalg.matmul(X, X) / 2

  def __call__(self, p_mean, p_covariance, delta, M):
    mu = p_mean + tf.linalg.matvec(p_covariance, delta)
    exp_mapping = ApproximateTractableCholeskyParametrization.exp_map(0.5 * M)
    A = tf.linalg.matmul(p_covariance, exp_mapping)
    
    return mu, tf.matmul(A, A, transpose_b=True)

  def inverse(self, mean, covariance):
    return mean, tf.linalg.cholesky(covariance)