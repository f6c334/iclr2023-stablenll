import numpy as np
import mpmath as mpm
import tensorflow as tf


# Derive the taylor and pade' coefficients for MTP, MPA
MAX_TAYLOR_COEFFICIENTS, MAX_PADE_COEFFICIENTS = 10, 5
taylor_coefficients = mpm.taylor(lambda x : mpm.sqrt(mpm.mpf(1) - x), 0, MAX_TAYLOR_COEFFICIENTS)
pade_p, pade_q = mpm.pade(taylor_coefficients, MAX_PADE_COEFFICIENTS, MAX_PADE_COEFFICIENTS)
pade_p = tf.constant(np.array(pade_p).astype(float), dtype=tf.keras.backend.floatx())
pade_q = tf.constant(np.array(pade_q).astype(float), dtype=tf.keras.backend.floatx())


def matrix_pade_approximant(p, I, K=5):
  p_sqrt, q_sqrt = pade_p[0] * I, pade_q[0] * I
  p_app = I - p
  p_hat = p_app

  for i in range(K):
    p_sqrt += pade_p[i + 1] * p_hat
    q_sqrt += pade_q[i + 1] * p_hat
    p_hat = tf.linalg.matmul(p_hat, p_app)
  
  return tf.linalg.solve(q_sqrt, p_sqrt)

def matrix_pade_approximant_inverse(p, I, K=5):
  p_sqrt, q_sqrt = pade_p[0] * I, pade_q[0]*I
  p_app = I - p
  p_hat = p_app

  for i in range(K):
    p_sqrt += pade_p[i+1] * p_hat
    q_sqrt += pade_q[i+1] * p_hat
    p_hat = tf.linalg.matmul(p_hat, p_app)

  return tf.linalg.solve(p_sqrt, q_sqrt)


# @tf.custom_gradient
def fast_pade_sqrtm(A: tf.Tensor):
  norm_A = tf.linalg.norm(A, ord='euclidean', axis=[-2, -1], keepdims=True)
  I = tf.eye(tf.shape(A)[-1], dtype=A.dtype)

  A_sqrt = matrix_pade_approximant(A / norm_A, I)
  A_sqrt = tf.sqrt(norm_A) * A_sqrt

  # def _custom_gradient(upstream):
  #   b = A_sqrt / tf.sqrt(norm_A)


    
  #   c = grad_output / torch.sqrt(normM)
  #   for i in range(8):
  #       #In case you might terminate the iteration by checking convergence
  #       #if th.norm(b-I)<1e-4:
  #       #    break
  #       b_2 = b.bmm(b)
  #       c = 0.5 * (c.bmm(3.0*I-b_2)-b_2.bmm(c)+b.bmm(c).bmm(b))
  #       b = 0.5 * b.bmm(3.0 * I - b_2)
  #   grad_input = 0.5 * c
  #   return grad_input
  return A_sqrt


def fast_pade_sqrtm_inv(A: tf.Tensor):
  norm_A = tf.linalg.norm(A, ord='euclidean', axis=[-2, -1], keepdims=True)
  I = tf.eye(tf.shape(A)[-1], dtype=A.dtype)

  A_sqrt_inv = matrix_pade_approximant_inverse(A / norm_A, I)
  A_sqrt_inv = A_sqrt_inv / tf.sqrt(norm_A)

  return A_sqrt_inv