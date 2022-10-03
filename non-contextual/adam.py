import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.common import sample_multivariate_normal



def adam_gaussian_optimization(initial_mean, initial_covariance, 
                              true_mean, true_covariance, 
                              phi, alpha, 
                              iterations, batch_size, mini_batches):
  @tf.function
  def _optimize(train_dataset, p_mean, p_covariance, phi, optimizer):
    for _, minibatch in enumerate(train_dataset):
      variables = [p_mean, p_covariance]
      
      # calculate loss
      with tf.GradientTape() as tape:
        mean, covariance = phi(p_mean, p_covariance)

        distribution = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(covariance))
        loss = - tf.reduce_mean(distribution.log_prob(minibatch))
        
      # calculate and apply gradients
      gradients = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))

  generate_dataset = lambda num_samples : sample_multivariate_normal(num_samples, true_mean, true_covariance)

  p_mean, p_covariance = phi.inverse(initial_mean, initial_covariance)
  p_mean, p_covariance = tf.Variable(p_mean), tf.Variable(p_covariance)
  optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)



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
  initial_mean, initial_covariance = np.zeros(shape=(10,)), np.diag(np.ones(shape=(10,)))
  true_mean, true_covariance = 4.0 * np.ones(shape=(10,)), 0.5 * np.diag(np.ones(shape=(10,)))

  phi = parametrizations.CholeskyParametrization()

  alpha = 5e-2
  iterations, batch_size, mini_batches = 400, 128, 8
  
  save_file = '10d_sample_run.npz'
  ######################

  ################### PLOTTING TEST
  log = adam_gaussian_optimization(initial_mean, initial_covariance,
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



"""NEXT THING
optimizers:
- trp
- pitfalls
- (lbfgs)

parametrizations:
- log diag parametrization
- natural parametrization
- square root parametrization (A AT, i.e. true sqrt)
"""