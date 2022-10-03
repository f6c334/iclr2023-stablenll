import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn import model_selection

from tr_projections.tensorflow import vectorized_trust_region_layers as vtrl

from common import utils, functions, hypertuning, callbacks, run_util
import models as m

NUM_FEATURES = NUM_OUTPUTS = NUM_GAUSS_DIM = 1
SF_FUNCTION_CONFIG = {
  'pitfalls': (lambda x: functions.sf_pitfalls_sinusoidal(x, offset=0.0, noise=0.1), (0.0, 12.0), 2000),
  'n_constant':
    (lambda x: functions.sf_n_constant(x, x_range=[-1.0, 1.0], y_range=[-3.0, 3.0], noises=[0.01, 0.1, 1.0, 10.0]),
     (-1.0, 1.0), 2000),
  'nn_constant': (lambda x: functions.sf_n_constant(
    x, x_range=[-1.0, 1.0], y_range=[-3.0, 3.0], noises=[0.01, 0.02, 1.0, 5.0, 0.01, 0.02, 1.0, 5.0]), (-1.0, 1.0), 2000
                 ),
  'detlefsen': (lambda x: functions.sf_detlefsen_sinusoidal(x, scale=1.0), (0.0, 10.0), 2000),
}

SHARED_MODEL_PARAMETERS = {
  'input_shape': [NUM_FEATURES],
  'n_dims': NUM_OUTPUTS,
  'gauss_dimension': NUM_GAUSS_DIM,
  'activations': 'relu',
  'cov_stab': (1e-8, 1e3)
}

SF_MODEL_CONFIG = {
  'AdamModel':
    lambda hparams, **kwargs: m.AdamModel(**SHARED_MODEL_PARAMETERS,
                                          learning_rate=hparams.Choice(name='learning_rate',
                                                                       values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                          **kwargs),
  'PitfallsModel':
    lambda hparams, **kwargs: m.PitfallsModel(**SHARED_MODEL_PARAMETERS,
                                              learning_rate=hparams.Choice(name='learning_rate',
                                                                           values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                              beta=0.5,
                                              **kwargs),
  'TractableModel':
    lambda hparams, **kwargs: m.TractableModel(**SHARED_MODEL_PARAMETERS,
                                               approximate_expm=True,
                                               mean_metric='mse',
                                               covariance_metric='mse',
                                               batch_reduce=tf.reduce_mean,
                                               learning_rate=hparams.Choice(name='learning_rate',
                                                                            values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                               beta=hparams.Choice(name='beta', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                               **kwargs),
  'TRPLW2Model':
    lambda hparams, **kwargs: m.TrueTrustRegionModel(
      **SHARED_MODEL_PARAMETERS,
      learning_rate=hparams.Choice(name='learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
      proj_layer=vtrl.W2ProjectionLayer(mean_bound=hparams.Choice(name='mean_bound', values=[1e1, np.inf]), covariance_bound=hparams.Choice(name='covariance_bound', values=[1e2, 1e1, 1e0, 1e-1])),
      **kwargs),
  'TrustableW2Model':
    lambda hparams, **kwargs: m.TraptableModel(
      **SHARED_MODEL_PARAMETERS,
      learning_rate=hparams.Choice(name='learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
      proj_layer=vtrl.W2ProjectionLayer(
        mean_bound=hparams.Choice(name='mean_bound', values=[1e1, np.inf]), covariance_bound=hparams.Choice(name='covariance_bound', values=[1e2, 1e1, 1e0, 1e-1])),
      beta=hparams.Choice(name='beta', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
      approximate_expm=True,
      project_natural_parameters=True,
      regress_on_projected_parameters=True,
      use_tractable_before_projection=False,
      mean_metric='mse',
      covariance_metric='w2',
      batch_reduce=tf.reduce_sum,
      advanced_metrics=False,
      **kwargs),
}





if __name__ == '__main__':
  import sys

  # physical_devices = tf.config.list_physical_devices('GPU')
  # tf.config.experimental.set_memory_growth(physical_devices[0], True)

  ROOT_DIR = 'logs'
  function_id, model_id, model_type = sys.argv[1:4]
  
  if model_type == 'simple_model':
    hidden_layers = [50, 50]
  elif model_type == 'complex_model':
    hidden_layers = [50, 50, 50]
  
  seed = 0
  utils.seeding(seed=seed, tf_deterministic=False)
  logdir = os.path.join(ROOT_DIR, 'simple_functions', model_type, function_id, model_id)

  # prepare data and dataset splits
  true_function_, function_range, function_samples = SF_FUNCTION_CONFIG[function_id]
  true_function = lambda x: functions.sample_wrapper(true_function_, x)

  X = np.linspace(*function_range, function_samples).reshape(-1, 1)
  Y, _, _ = true_function(X)
  Y = Y[..., np.newaxis]

  build_model = lambda hparams, **kwargs: SF_MODEL_CONFIG[model_id](hparams, hidden_layers=hidden_layers, **kwargs)
  
  run_util.basic_regression_run(logdir,
                                build_model,
                                true_function,
                                X,
                                Y,
                                save_model=False,
                                normalize_targets=True,
                                test_size=0.3,
                                val_size=0.3,
                                hypertuning_epochs=1000,
                                early_stopping_patience=200,
                                executions_per_trial=2,
                                max_trials=1000,
                                num_best_model_runs=10,
                                best_model_epochs=3000,
                                batch_size=512,
                                graph_callback=None,
                                plot_frequency=None,
                                weight_histogram_frequency=0,
                                skip_hypertuning=False,
                                load_tuner_only=False,
                                add_additional_metrics=True,
                                seed=seed,
                                verbose=0)