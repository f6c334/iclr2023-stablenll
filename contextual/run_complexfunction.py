import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle

import keras_tuner as kt
import matplotlib

matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

from sklearn import model_selection

from tr_projections.tensorflow import vectorized_trust_region_layers as vtrl

from common import utils, functions, hypertuning, callbacks, run_util
import models as m

NUM_FEATURES = NUM_OUTPUTS = 1

CF_FUNCTION_CONFIG = {
  '3dspiral': (lambda x: functions.cf_function1(x), (0.0, 10.0), 5000),
  '3dq2spiral': (lambda x: functions.cf_function2(x), (0.0, 20.0), 5000),
  '3dspiral_sq': (lambda x: functions.cf_function2(x), (0.0, 20.0), 5000),
  '3dspiral_sm4': (lambda x: functions.cf_function2(x, cov_scale=1e-4), (0.0, 20.0), 5000),
  '3dspiral_sm5': (lambda x: functions.cf_function2(x, cov_scale=1e-5), (0.0, 20.0), 5000),  # 1e0
}

SHARED_MODEL_PARAMETERS = {
  'input_shape': [NUM_FEATURES],
  'n_dims': NUM_OUTPUTS,
  'gauss_dimension': 3,
  'hidden_layers': [50, 50, 50],
  'activations': 'relu',
  'covariance_head_type': 'cholesky'
}

CF_MODEL_CONFIG = {
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
                                               covariance_metric='w2',
                                               batch_reduce=tf.reduce_mean,
                                               learning_rate=hparams.Choice(name='learning_rate',
                                                                            values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                               beta=hparams.Choice(name='beta', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                               **kwargs),
  'TRPLW2Model':
    lambda hparams, **kwargs: m.TrueTrustRegionModel(
      **SHARED_MODEL_PARAMETERS,
      learning_rate=hparams.Choice(name='learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
      proj_layer=vtrl.W2ProjectionLayer(mean_bound=1e1, covariance_bound=1e1),
      **kwargs),
  'TrustableW2Model':
    lambda hparams, **kwargs: m.TraptableModel(
      **SHARED_MODEL_PARAMETERS,
      learning_rate=hparams.Choice(name='learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
      proj_layer=vtrl.W2ProjectionLayer(mean_bound=np.inf, covariance_bound=1e1),
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
  function_id, model_id = sys.argv[1:3]
  
  seed = 0
  utils.seeding(seed=seed, tf_deterministic=True)
  logdir = os.path.join(ROOT_DIR, 'complex_functions', function_id, model_id)


  # prepare data and dataset splits
  true_function_, function_range, function_samples = CF_FUNCTION_CONFIG[function_id]
  true_function = lambda x: functions.sample_wrapper(true_function_, x)

  X = np.linspace(*function_range, function_samples)
  Y, _, _ = true_function(X)
  X, Y = X.reshape(-1, 1), np.expand_dims(Y, axis=-2)

  build_model = CF_MODEL_CONFIG[model_id]

  run_util.basic_regression_run(logdir,
                                build_model,
                                true_function,
                                X,
                                Y,
                                save_model=False,
                                normalize_targets=False,
                                test_size=0.3,
                                val_size=0.3,
                                hypertuning_epochs=1000,
                                num_best_model_runs=10,
                                best_model_epochs=3000,
                                batch_size=512,
                                graph_callback=callbacks.Graph3DCallback,
                                plot_frequency=50,
                                skip_hypertuning=False,
                                load_tuner_only=False,
                                add_additional_metrics=True,
                                seed=seed)