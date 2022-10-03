import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.common import sample_multivariate_normal






def tractable_gaussian_optimization(initial_mean, initial_covariance, 
                              true_mean, true_covariance, 
                              phi, alpha, 
                              iterations, batch_size, mini_batches):
  @tf.function
  def _optimize(train_dataset, p_mean, p_covariance, delta, M, phi, alpha):
    for _, minibatch in enumerate(train_dataset):
      
      delta.assign(tf.zeros_like(p_mean))
      M.assign(tf.zeros_like(p_covariance))

      # calculate loss
      with tf.GradientTape() as tape:
        mean, covariance = phi(p_mean, p_covariance, delta, M)

        distribution = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(covariance))
        loss = - tf.reduce_mean(distribution.log_prob(minibatch))
        
      # calculate and apply gradients
      [d_delta, d_M] = tape.gradient(loss, [delta, M])

      new_mean, new_covariance = phi(p_mean, p_covariance, - alpha * d_delta, - alpha * d_M)
      new_p_mean, new_p_covariance = phi.inverse(new_mean, new_covariance)

      p_mean.assign(new_p_mean)
      p_covariance.assign(new_p_covariance)

  generate_dataset = lambda num_samples : sample_multivariate_normal(num_samples, true_mean, true_covariance)
  
  p_mean, p_covariance = phi.inverse(initial_mean, initial_covariance)
  p_mean, p_covariance = tf.Variable(p_mean), tf.Variable(p_covariance)

  # logging functions
  log = []
  log_data = lambda log_: log.append(log_ | {'t' : time.time_ns()})

  mean, covariance = phi(p_mean, p_covariance, tf.zeros_like(p_mean), tf.zeros_like(p_covariance))
  log_data({'i' : -1, 'mean' : mean.numpy(), 'covariance' : covariance.numpy()})


  for i in range(iterations):
    
    # generate samples
    train_dataset = generate_dataset(num_samples=batch_size * mini_batches).batch(batch_size).take(mini_batches)

    delta = tf.Variable(tf.zeros_like(p_mean), trainable=True)
    M = tf.Variable(tf.zeros_like(p_covariance), trainable=True)
    _optimize(train_dataset, p_mean, p_covariance, delta, M, phi, alpha)
    
    mean, covariance = phi(p_mean, p_covariance, tf.zeros_like(p_mean), tf.zeros_like(p_covariance))
    log_data({'i' : i, 'mean' : mean.numpy(), 'covariance' : covariance.numpy()})

    if i % 50 == 0:
      from tabulate import tabulate

      # tabulate data
      mean_table = tabulate(mean.numpy().reshape(1, -1), tablefmt="fancy_grid", floatfmt=".2f")
      cov_table = tabulate(covariance.numpy(), tablefmt="fancy_grid", floatfmt=".2f")
      
      print(f'[{i}] #################################\nmean : \n{mean_table}\ncovariance : \n{cov_table}')
  
  return log


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import parametrizations
  from utils.plotting import nd_gaussian_visualization_plot

  ### HYPERPARAMS ######
  #initial_mean, initial_covariance = 0.0 * np.ones(shape=(2,)), np.asarray([[0.92378204, -0.77616199], [-0.77616199, 1.734638]])
  initial_mean, initial_covariance = 0.0 * np.ones(shape=(2,)), np.asarray([[88.56752592, -37.82297329], [-37.82297329, 16.16368182]])
  #initial_mean, initial_covariance = 0.0 * np.ones(shape=(2,)), np.asarray([[1.0, -0.5], [-0.5, 1.0]])
  true_mean, true_covariance = 5.0 * np.ones(shape=(2,)), np.asarray([[1.0, 0.5], [0.5, 1.0]])

  phi = parametrizations.TractableCholeskyParametrization()

  alpha = 1e-2
  iterations, batch_size, mini_batches = 400, 128, 8
  
  save_file = '10d_sample_run.npz'
  ######################

  ################### PLOTTING TEST
  log = tractable_gaussian_optimization(initial_mean, initial_covariance,
                                  true_mean, true_covariance,
                                  phi, alpha,
                                  iterations, batch_size, mini_batches)

  means = np.asarray([log_['mean'] for log_ in log])
  covariances = np.asarray([log_['covariance'] for log_ in log])

  if save_file is not None:
    np.savez(save_file, 
            means=means, covariances=covariances, 
            true_mean=true_mean, true_covariance=true_covariance)

  nd_gaussian_visualization_plot(means, covariances, true_mean, true_covariance, num_means_parallel_coords=10, num_covariances_heatmaps=6)
  plt.tight_layout()
  plt.show()
  #################################
