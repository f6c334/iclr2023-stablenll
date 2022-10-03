import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.common import sample_multivariate_normal

import tr_projections.tensorflow.trust_region_layers as trust_region_layers




def trp_gaussian_optimization(initial_mean, initial_covariance, 
                              true_mean, true_covariance, 
                              phi, alpha, proj_layer,
                              iterations, batch_size, mini_batches):
  #@tf.function
  def _optimize(train_dataset, p_mean, p_covariance, phi, optimizer, mean_old, covariance_old, proj_layer):
    for _, minibatch in enumerate(train_dataset):
        
      # calculate loss
      with tf.GradientTape() as tape:
        mean, covariance = phi(p_mean, p_covariance)
        
        if isinstance(proj_layer, trust_region_layers.W2ProjectionLayer):
          sqrt_covariance, sqrt_covariance_old = tf.linalg.sqrtm(covariance), tf.linalg.sqrtm(covariance_old)
          proj_mean, sqrt_proj_covariance = proj_layer(mean, mean_old, sqrt_covariance, sqrt_covariance_old)
          proj_covariance = tf.matmul(sqrt_proj_covariance, sqrt_proj_covariance)
        else:
          proj_mean, proj_covariance = proj_layer(mean, mean_old, covariance, covariance_old)

        distribution = tfd.MultivariateNormalTriL(loc=proj_mean, scale_tril=tf.linalg.cholesky(proj_covariance))
        loss = - tf.reduce_mean(distribution.log_prob(minibatch))
      
      # calculate and apply gradients
      variables = [p_mean, p_covariance]
      gradients = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))

      # now update estimation to projection
      mean, covariance = phi(p_mean, p_covariance)
      proj_mean, proj_covariance = proj_layer(mean, mean_old, covariance, covariance_old)
      p_proj_mean, p_proj_covariance = phi.inverse(proj_mean, proj_covariance)
      
      p_mean.assign(p_proj_mean)
      p_covariance.assign(p_proj_covariance)

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

    mean_old, covariance_old = tf.identity(mean), tf.identity(covariance)

    _optimize(train_dataset, p_mean, p_covariance, phi, optimizer, mean_old, covariance_old, proj_layer)
    
    mean, covariance = phi(p_mean, p_covariance)
    log_data({'i' : i, 'mean' : mean.numpy(), 'covariance' : covariance.numpy()})

    if i % 50 == 0:
      from tabulate import tabulate

      # tabulate data
      mean_table = tabulate(mean.numpy().reshape(1, -1), tablefmt="fancy_grid", floatfmt=".2f")
      cov_table = tabulate(covariance.numpy(), tablefmt="fancy_grid", floatfmt=".2f")
      
      print(f'[{i}] #################################\nmean : \n{mean_table}\ncovariance : \n{cov_table}')

    if tf.math.is_nan(mean).numpy().any() or tf.math.is_nan(covariance).numpy().any():
      for j in range(i + 1, iterations):
        log_data({'i' : j, 'mean' : mean.numpy(), 'covariance' : covariance.numpy()})
      return log

  return log



if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import parametrizations
  from utils.plotting import nd_gaussian_visualization_plot, plot_nd_gaussian_kl

  ### GENERATE PARAMS ###
  from utils.common import rotation_matrix_around_standard_spans, seeding
  
  SEED = 0
  seeding(SEED, tf_deterministic=False)

  SAMPLES_PER_CASE = 10
  def random_covariance(eigenvalues):
    rotations = np.random.uniform(low=0, high=np.pi, size=eigenvalues.shape[0] - 1)
    R = rotation_matrix_around_standard_spans(rotations)
    return R @ np.diag(eigenvalues) @ R.T

  def random_mean(low, high, size):
    return np.random.uniform(low, high, size)

  optimal_eigval_dist = lambda size: np.random.beta(2, 2, size) + 0.5     # [0.5, 1.5], concentrated on 1
  small_eigval_dist = lambda size: np.random.beta(0.5, 8, size) + 0.01    # [0.01, 1.01], concentrated on 0.01
  large_eigval_dist = lambda size: np.random.beta(8, 0.5, size) * 100.0   # [0, 100.0], concentrated on 100
  inoptimal_eigval_dist = lambda size: np.random.choice(np.concatenate([small_eigval_dist(size), large_eigval_dist(size)]), size)   # randomly sample from each


  # which configs would be interesting to compute?
  case_configs = [
    # 1. optimal - optimal    : expect good convergence behaviour
    [{  'initial_mean' : random_mean(low=-5, high=5, size=(10)),    'initial_covariance' : random_covariance(optimal_eigval_dist(size=(10))),
        'true_mean' : random_mean(low=-5, high=5, size=(10)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(10)))     }
      for _ in range(SAMPLES_PER_CASE)],
    # 2. small - optimal      : expect explosion at the beginning
    [{  'initial_mean' : random_mean(low=-5, high=5, size=(10)),    'initial_covariance' : random_covariance(small_eigval_dist(size=(10))),
        'true_mean' : random_mean(low=-5, high=5, size=(10)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(10)))     }
      for _ in range(SAMPLES_PER_CASE)],
    # 3. large - optimal      : expect very slow convergence
    [{  'initial_mean' : random_mean(low=-5, high=5, size=(10)),    'initial_covariance' : random_covariance(large_eigval_dist(size=(10))),
        'true_mean' : random_mean(low=-5, high=5, size=(10)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(10)))     }
      for _ in range(SAMPLES_PER_CASE)],
    # 4. inoptimal - optimal  : expect jumps at beginning, then slow convergence
    [{  'initial_mean' : random_mean(low=-5, high=5, size=(10)),    'initial_covariance' : random_covariance(small_eigval_dist(size=(10))),
        'true_mean' : random_mean(low=-5, high=5, size=(10)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(10)))     }
      for _ in range(SAMPLES_PER_CASE)],
  ]

  ### HYPERPARAMS ######
  normal_config = case_configs[2][5]
  initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()

  phi = parametrizations.SqrtCovarianceParametrization()
  proj_layer = trust_region_layers.W2ForceProjectionLayer(mean_bound=0.1, covariance_bound=0.1)

  alpha = 5e-2
  iterations, batch_size, mini_batches = 200, 128, 8
  
  save_file = '10d_sample_run.npz'
  ######################

  ################### PLOTTING TEST
  seeding(SEED, tf_deterministic=False)
  log = trp_gaussian_optimization(initial_mean, initial_covariance,
                                  true_mean, true_covariance,
                                  phi, alpha, proj_layer,
                                  iterations, batch_size, mini_batches)

  means = np.asarray([log_['mean'] for log_ in log])
  covariances = np.asarray([log_['covariance'] for log_ in log])
  times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9   # in seconds

  if save_file is not None:
    np.savez(save_file, 
            means=means, covariances=covariances, 
            true_mean=true_mean, true_covariance=true_covariance)

  fig, ax = plt.subplots(1, 1)
  plot_nd_gaussian_kl(means, covariances,
                      true_mean, true_covariance,
                      label='Parametrization',
                      ax=ax,
                      times=times,
                      normalize_kl=False,
                      log_scale=True)
  
  plt.tight_layout()
  plt.show()
  #################################
