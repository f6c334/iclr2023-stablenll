import gc
import os
import pickle

import tensorflow as tf

from sklearn import model_selection

from common import callbacks, hypertuning, utils



def basic_regression_run(logdir,
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
                         max_trials=100,
                         num_best_model_runs=10,
                         best_model_epochs=3000,
                         batch_size=256,
                         graph_callback=None,
                         plot_frequency=100,
                         weight_histogram_frequency=0,
                         skip_hypertuning=False,
                         load_tuner_only=False,
                         add_additional_metrics=True,
                         seed=0,
                         verbose=3):
  # prepare dataset for hyperparameter optimization and training
  (X_train, Y_train), (X_test, Y_test), (X_mean, X_std, Y_mean,
                                         Y_std) = utils.split_normalize_data(X,
                                                                             Y,
                                                                             test_size=test_size,
                                                                             normalize_targets=normalize_targets,
                                                                             seed=seed)
  X_train_, X_val_, Y_train_, Y_val_ = model_selection.train_test_split(X_train,
                                                                        Y_train,
                                                                        test_size=val_size,
                                                                        random_state=seed)
  target_denormalizer = (lambda X: utils.denormalize(X, Y_mean, Y_std)) if normalize_targets else None

  # model and hyperparameter space definition
  build_model_ = lambda hparams: build_model(hparams, target_denormalizer=target_denormalizer)

  # tune for best hyperparameters on separate validation set
  if not skip_hypertuning:
    tuner = hypertuning.sf_hypertune(logdir=logdir,
                                     build_model=build_model_,
                                     training_data=(X_train_, Y_train_),
                                     validation_data=(X_val_, Y_val_),
                                     max_trials=max_trials,
                                     executions_per_trial=executions_per_trial,
                                     overwrite=False,
                                     max_epochs=hypertuning_epochs,
                                     early_stopping_patience=early_stopping_patience,
                                     batch_size=batch_size,
                                     load_tuner_only=load_tuner_only,
                                     verbose=verbose,
                                     seed=seed)

  # now tuner get best model and train again
  for i in range(num_best_model_runs):   
    if skip_hypertuning:
      best_model = build_model_(None)
    else:
      best_model = build_model_(tuner.get_best_hyperparameters()[0])

    # best_model = build_model(None)
    best_model_logdir = os.path.join(logdir, 'best_model', f'run_{i}')

    callbacks_ = [
      tf.keras.callbacks.TensorBoard(best_model_logdir, histogram_freq=weight_histogram_frequency, write_graph=False),
      tf.keras.callbacks.EarlyStopping(monitor='val_losses/unnormalized_nll', patience=best_model_epochs, restore_best_weights=True),
      # callbacks.MemoryUsageCallback()
    ]

    if graph_callback is not None:
      callbacks_ += [
        graph_callback(training_data=(X_train, Y_train),
                       validation_data=(X_test, Y_test),
                       input_denormalizer=lambda X: utils.denormalize(X, X_mean, X_std),
                       target_denormalizer=target_denormalizer,
                       true_function=true_function,
                       plot_frequency=plot_frequency,
                       logdir=best_model_logdir),
      ]
    if add_additional_metrics:
      callbacks_ += [
        callbacks.AdditionalMetricsWithTrueFunctionCallback(
          validation_data=(X_test, Y_test),
          input_denormalizer=lambda X: utils.denormalize(X, X_mean, X_std),
          target_denormalizer=target_denormalizer,
          true_function=true_function)
      ]

    history = best_model.fit(X_train,
                             Y_train,
                             epochs=best_model_epochs,
                             batch_size=batch_size,
                             validation_data=(X_test, Y_test),
                             callbacks=callbacks_,
                             verbose=verbose)

    if save_model:
      best_model.save(os.path.join(best_model_logdir, f'model'),
                      overwrite=True,
                      include_optimizer=False,
                      save_format='tf')
    with open(os.path.join(logdir, f'history_bm_{i}.pkl'), 'wb') as file:
      pickle.dump(history.history, file)
    
    # bookkeeping
    tf.keras.backend.clear_session()
    del best_model
    gc.collect()

def uci_regression_run(logdir,
                       build_model,
                       X,
                       Y,
                       save_model=False,
                       normalize_targets=True,
                       test_size=0.3,
                       val_size=0.3,
                       num_splits=20,
                       hypertuning_epochs=1000,
                       early_stopping_patience=200,
                       executions_per_trial=1,
                       num_best_model_runs=10,
                       best_model_epochs=3000,
                       batch_size=256,
                       weight_histogram_frequency=0,
                       skip_hypertuning=False,
                       load_tuner_only=False,
                       seed=0):
  for i in range(num_splits):
    basic_regression_run(
      logdir=os.path.join(logdir, f'split_{i}'),
      build_model=build_model,
      true_function=None,
      X=X,
      Y=Y,
      save_model=save_model,
      normalize_targets=normalize_targets,
      test_size=test_size,
      val_size=val_size,
      hypertuning_epochs=hypertuning_epochs,
      early_stopping_patience=early_stopping_patience,
      executions_per_trial=executions_per_trial,
      num_best_model_runs=num_best_model_runs,
      best_model_epochs=best_model_epochs,
      batch_size=batch_size,
      graph_callback=None,
      plot_frequency=0,
      weight_histogram_frequency=weight_histogram_frequency,
      skip_hypertuning=skip_hypertuning,
      load_tuner_only=load_tuner_only,
      add_additional_metrics=False,
      seed=seed + i
    )