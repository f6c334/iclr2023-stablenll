import os

import keras_tuner as kt
import tensorflow as tf
from sklearn import model_selection

from common import utils

from common import gridtuner

def sf_train_test(logdir,
                  build_model,
                  features,
                  labels,
                  max_updates=20000,
                  batch_size=256,
                  seed=0,
                  test_size=0.2,
                  val_size=0.2):
  # prepare data split and normalize based on full training set
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features,
                                                                      labels,
                                                                      test_size=test_size,
                                                                      random_state=seed)
  X_train, X_train_mean, X_train_std = utils.normalize(X_train)
  Y_train, Y_train_mean, Y_train_std = utils.normalize(Y_train)
  X_test_normalized, _, _ = utils.normalize(X_test, X_train_mean, X_train_std)
  Y_test_normalized, _, _ = utils.normalize(Y_test, Y_train_mean, Y_train_std)

  X_train_, X_val_, Y_train_, Y_val_ = model_selection.train_test_split(X_train,
                                                                        Y_train,
                                                                        test_size=val_size,
                                                                        random_state=seed)

  build_model_ = lambda hparams: build_model(
      hparams, target_denormalizer=lambda X: utils.denormalize(X, Y_train_mean, Y_train_std))

  # search best hyperparameters with keras tuner
  tuner = kt.RandomSearch(
      hypermodel=build_model_,
      objective=kt.Objective('val_nll_loss', direction='min'),
      max_trials=10,
      executions_per_trial=1,
      overwrite=True,
      directory=logdir,
      project_name='project',
  )
  tuner.search_space_summary()
  tuner.search(x=X_train_,
               y=Y_train_,
               validation_data=(X_val_, Y_val_),
               epochs=max_updates,
               batch_size=batch_size,
               callbacks=[
                   tf.keras.callbacks.TensorBoard(logdir, write_graph=False),
                   tf.keras.callbacks.EarlyStopping(monitor='val_nll_loss', patience=50, restore_best_weights=True)
               ],
               verbose=3)

  # fit best model on whole train set
  best_model = build_model_(tuner.get_best_hyperparameters()[0])
  t_start = tf.timestamp()
  best_model.fit(X_train,
                 Y_train,
                 epochs=max_updates,
                 batch_size=batch_size,
                 validation_data=(X_test_normalized, Y_test_normalized),
                 callbacks=[
                     tf.keras.callbacks.TensorBoard(os.path.join(logdir, 'best_model')),
                     tf.keras.callbacks.EarlyStopping(monitor='val_nll_loss', patience=50, restore_best_weights=True)
                 ],
                 verbose=3)
  t_training = tf.timestamp() - t_start
  return t_training, best_model.evaluate(X_test_normalized, Y_test_normalized)


def sf_hypertune(logdir,
                 build_model,
                 training_data,
                 validation_data,
                 max_trials=10,
                 executions_per_trial=1,
                 overwrite=True,
                 max_epochs=5000,
                 early_stopping_patience=50,
                 batch_size=256,
                 load_tuner_only=False,
                 verbose=3,
                 seed=0):
  # search best hyperparameters with keras tuner
  tuner = gridtuner.GridSearch(
      hypermodel=build_model,
      objective=kt.Objective('val_losses/unnormalized_nll', direction='min'),
      max_trials=max_trials,
      executions_per_trial=executions_per_trial,
      overwrite=overwrite,
      directory=logdir,
      project_name='project',
      seed=seed
  )

  tuner.search_space_summary()
  
  if not load_tuner_only:
    tuner.search(*training_data,
                validation_data=validation_data,
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(logdir, write_graph=False),
                    tf.keras.callbacks.EarlyStopping(monitor='val_losses/unnormalized_nll',
                                                      patience=early_stopping_patience,
                                                      restore_best_weights=True)
                ],
                verbose=verbose)
  return tuner