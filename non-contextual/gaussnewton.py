import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.common import sample_multivariate_normal





def gaussnewton_gaussian_optimization(initial_mean, initial_covariance, 
                              true_mean, true_covariance, 
                              phi, alpha, 
                              iterations, batch_size, mini_batches):
  @tf.function
  def _optimize(train_dataset, p_mean, p_covariance, phi, optimizer):
    for _, minibatch in enumerate(train_dataset):
      variables = [p_mean, p_covariance]

      # calculate loss
      with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape(persistent=True) as tape_:
          mean, covariance = phi(p_mean, p_covariance)

          distribution = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(covariance))
          loss = - tf.reduce_mean(distribution.log_prob(minibatch))
        gradients = tape_.gradient(loss, variables)
      hessians = [tape.jacobian(gradient, variable) for gradient, variable in zip(gradients, variables)]

      gradients = [tf.reshape(gradient, shape=(-1, 1)) for gradient in gradients]
      hessians = [tf.reshape(hessian, shape=(gradient.shape[0], gradient.shape[0])) for gradient, hessian in zip(gradients, hessians)]

      newton_gradients = [tf.linalg.lstsq(hessian, gradient, l2_regularizer=1e-7) for gradient, hessian in zip(gradients, hessians)]
      newton_gradients = [tf.reshape(gradient, shape=variable.shape) for gradient, variable in zip(newton_gradients, variables)]
        
      # calculate and apply hessians
      optimizer.apply_gradients(zip(newton_gradients, variables))

  generate_dataset = lambda num_samples : sample_multivariate_normal(num_samples, true_mean, true_covariance)

  p_mean, p_covariance = phi.inverse(initial_mean, initial_covariance)
  p_mean, p_covariance = tf.Variable(p_mean), tf.Variable(p_covariance)
  optimizer = tf.keras.optimizers.SGD(learning_rate=alpha)

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

    if i % 10 == 0:
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
  initial_mean, initial_covariance = 0.0 * np.ones(shape=(1,)), 1.0 * np.diag(np.ones(shape=(1,)))
  true_mean, true_covariance = 5.0 * np.ones(shape=(1,)), 0.01 * np.diag(np.ones(shape=(1,)))

  phi = parametrizations.VanillaParametrization()

  alpha = 0.1
  iterations, batch_size, mini_batches = 25, 128, 8
  
  save_file = '10d_sample_run.npz'
  ######################

  ################### PLOTTING TEST
  log = gaussnewton_gaussian_optimization(initial_mean, initial_covariance,
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