import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle

import keras_tuner as kt
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

from sklearn import model_selection

from tr_projections.tensorflow import vectorized_trust_region_layers as vtrl

from common import utils, functions, hypertuning, callbacks, run_util
from uci_datasets import uci
import models as m


SHARED_MODEL_PARAMETERS = {
  'n_dims': 1,
  'hidden_layers': [50, 50],
  'activations': 'relu',
  'covariance_head_type': 'cholesky',
  'cov_stab': (1e-3, 1e3)
}

MODEL_CONFIG = {
  'AdamModel':
    lambda hparams, **kwargs: m.AdamModel(**SHARED_MODEL_PARAMETERS,
                                          learning_rate=hparams.Choice(name='learning_rate',
                                                                       values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                          **kwargs),
  'Pitfalls05Model':
    lambda hparams, **kwargs: m.PitfallsModel(**SHARED_MODEL_PARAMETERS,
                                              learning_rate=hparams.Choice(name='learning_rate',
                                                                           values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                              beta=0.5,
                                              **kwargs),
  'Pitfalls10Model':
    lambda hparams, **kwargs: m.PitfallsModel(**SHARED_MODEL_PARAMETERS,
                                              learning_rate=hparams.Choice(name='learning_rate',
                                                                           values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
                                              beta=1.0,
                                              **kwargs),
  'TrustableW2Model':
    lambda hparams, **kwargs: m.TraptableModel(**SHARED_MODEL_PARAMETERS,
                                               learning_rate=hparams.Choice(name='learning_rate',
                                                                            values=[1e-3, 5e-3, 1e-2]),
                                               proj_layer=vtrl.W2ProjectionLayer(mean_bound=hparams.Choice(name='mean_bound', values=[1e-1, np.inf]), covariance_bound=hparams.Choice(name='covariance_bound', values=[1e-4, 1e-1])),
                                               beta=hparams.Choice(name='beta', values=[1e-4, 1e-3, 1e-2]),
                                               local_covariance_gradient_scale=2.0,
                                               approximate_expm=True,
                                               project_natural_parameters=True,
                                               regress_on_projected_parameters=True,
                                               use_tractable_before_projection=False,
                                               mean_metric='mse',
                                               covariance_metric='w2_comm',
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
  utils.seeding(seed=seed, tf_deterministic=False)
  logdir = os.path.join(ROOT_DIR, 'uci_multivariate', function_id, model_id)
  
  # prepare data
  X, Y, (num_samples, num_features, num_outputs) = uci.get(set_id=function_id)
  Y = np.expand_dims(Y, axis=-2)
  
  build_model = lambda hparams, **kwargs: MODEL_CONFIG[model_id](hparams, input_shape=[num_features], gauss_dimension=num_outputs, **kwargs)

  run_util.basic_regression_run(logdir,
                                build_model,
                                None,
                                X,
                                Y,
                                save_model=False,
                                normalize_targets=True,
                                test_size=0.2,
                                val_size=0.2,
                                hypertuning_epochs=2000,
                                early_stopping_patience=50,
                                executions_per_trial=2,
                                max_trials=1000,
                                num_best_model_runs=10,
                                best_model_epochs=5000,
                                batch_size=512,
                                graph_callback=None,
                                plot_frequency=None,
                                weight_histogram_frequency=0,
                                skip_hypertuning=False,
                                load_tuner_only=False,
                                add_additional_metrics=False,
                                seed=seed,
                                verbose=0)