import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.common import sample_multivariate_normal


def natural_gaussian_optimization(initial_mean, initial_covariance, 
                              true_mean, true_covariance, 
                              phi, delta,
                              iterations, batch_size, mini_batches):
  @tf.function
  def _optimize(train_dataset, p_mean, p_covariance, phi, optimizer):
    for _, minibatch in enumerate(train_dataset):
      variables = [p_mean, p_covariance]
      
      # calculate loss
      with tf.GradientTape(persistent=True) as tape:
        mean, covariance = phi(p_mean, p_covariance)

        distribution = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(covariance))
        log_pi = distribution.log_prob(minibatch)
        
      # estimate fisher
      dlogpi_variables = tape.jacobian(log_pi, variables, experimental_use_pfor=True)
      dlogpi = tf.concat([tf.reshape(x, shape=(batch_size, -1)) for x in dlogpi_variables], axis=1)
      
      fisher = dlogpi[..., None] * dlogpi[:, None] # elem-wise outer product
      fisher = tf.math.reduce_mean(fisher, axis=0)
      
      fisher_inv = tf.linalg.pinv(fisher) # + 1e-3 * tf.eye(tf.shape(fisher)[0], dtype=fisher.dtype))
        
      # mean and variance NLL gradient
      dvariables = [- tf.reduce_mean(dlogpi, axis=0) for dlogpi in dlogpi_variables]
        
      # natural gradient
      gradient = tf.concat([tf.reshape(x, shape=(-1,)) for x in dvariables], axis=0)
      natural_gradient = tf.linalg.matvec(fisher_inv, gradient)
      alpha = tf.sqrt((2 * delta) / tf.reduce_sum(tf.multiply(gradient, natural_gradient)))

      # applying natural gradient
      dp_mean, dp_covariance = natural_gradient[:tf.size(p_mean)], natural_gradient[tf.size(p_mean):]
      dp_mean = tf.reshape(dp_mean, shape=p_mean.shape)
      dp_covariance = tf.reshape(dp_covariance, shape=p_covariance.shape)
        
      # calculate and apply gradients
      optimizer.apply_gradients(zip([alpha * dp_mean, alpha * dp_covariance], variables))

  generate_dataset = lambda num_samples : sample_multivariate_normal(num_samples, true_mean, true_covariance)

  p_mean, p_covariance = phi.inverse(initial_mean, initial_covariance)
  p_mean, p_covariance = tf.Variable(p_mean), tf.Variable(p_covariance)
  optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

  # logging functions
  log = []
  log_data = lambda log_: log.append(log_ | {'t' : time.time_ns()})

  mean, covariance = phi(p_mean, p_covariance)
  log_data({'i' : -1, 'mean' : mean.numpy(), 'covariance' : covariance.numpy()})


  for i in range(iterations):
    
    # generate samples
    train_dataset = generate_dataset(num_samples=batch_size * mini_batches).batch(batch_size).take(mini_batches)

    _optimize(train_dataset, p_mean, p_covariance, phi, optimizer)

    mean, covariance = phi(p_mean, p_covariance)
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
  initial_mean, initial_covariance = np.zeros(shape=(2,)), np.diag(np.ones(shape=(2,)))
  true_mean, true_covariance = 5.0 * np.ones(shape=(2,)), 0.5 * np.diag(np.ones(shape=(2,)))

  phi = parametrizations.CholeskyParametrization()

  delta = 0.01
  iterations, batch_size, mini_batches = 120, 128, 8
  
  save_file = '10d_sample_run.npz'
  ######################

  ################### PLOTTING TEST
  log = natural_gaussian_optimization(initial_mean, initial_covariance,
                                  true_mean, true_covariance,
                                  phi, delta,
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



