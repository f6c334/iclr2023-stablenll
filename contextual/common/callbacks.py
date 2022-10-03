from enum import unique
import glob
import os
import psutil
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import plotting, utils


class Graph1DCallback(tf.keras.callbacks.Callback):

  def __init__(self,
               training_data=None,
               validation_data=None,
               input_denormalizer=None,
               target_denormalizer=None,
               true_function=None,
               plot_frequency=100,
               logdir=None):
    super(Graph1DCallback, self).__init__()
    self.training_data = training_data
    self.validation_data = validation_data
    self.plot_frequency = plot_frequency
    self.input_denormalizer = input_denormalizer
    self.target_denormalizer = target_denormalizer
    self.true_function = true_function
    self.logdir = logdir
    self.graphdir = os.path.join(logdir, 'eval_graphs')

    os.makedirs(self.graphdir, exist_ok=True)
    self.fig_function, self.ax_function = plt.subplots()
    self.fig_histogram, self.ax_histogram = plt.subplots()
    self.ax_histogram_twin = self.ax_histogram.twinx()

  def on_epoch_end(self, epoch, logs=None):
    if self.validation_data is None or epoch % self.plot_frequency != 0:
      return

    x, y = self.validation_data
    mean, covariance, A = self.model(x)
    

    # add projection layer if existent
    if hasattr(self.model, 'proj_layer') and hasattr(self.model, 'old_val_means'):
      mean, covariance = self.model.proj_layer(mean, self.model.old_val_means, covariance,
                                               self.model.old_val_covariances)

    y_unnormalized, mean_unnormalized, covariance_unnormalized = y, mean, covariance
    # denormalize input/output for plot
    if self.input_denormalizer is not None:
      x = self.input_denormalizer(X=x)

    if self.target_denormalizer is not None:
      y = self.target_denormalizer(X=y)
      mean, covariance = self.target_denormalizer(X=(mean, covariance))


    idx = np.argsort(x.ravel())
    x_, y_ = x.ravel()[idx], y.ravel()[idx]

    pred_mean, pred_std = mean.numpy().ravel()[idx], np.sqrt(covariance.numpy().ravel()[idx])
    # y_un = y_unnormalized.ravel()[idx]
    # pred_mean_un = mean_unnormalized.numpy().ravel()[idx]
    # pred_std_un = np.sqrt(covariance_unnormalized.numpy().ravel()[idx])

    nll = 0.5 * (np.log(2.0 * np.pi) + 2 * np.log(pred_std) + np.square((y_ - pred_mean) / pred_std))
    # nll_un = 0.5 * (np.log(2.0 * np.pi) + 2 * np.log(pred_std_un) + np.square((y_un - pred_mean_un) / pred_std_un))

    # plot function
    self._plot_function(x_, pred_mean, pred_std, ax=self.ax_function)

    self.fig_function.legend()
    self.fig_function.suptitle(
      f'Prediction Epoch {epoch}\nNLL:{logs["val_nll_loss"]:6.3f} | MSE:{logs["val_mse_mean"]:6.3f} ')
    self.fig_function.set_tight_layout(True)

    # TODO: plot histogram
    self._plot_function(x_,
                        pred_mean,
                        pred_std,
                        ax=self.ax_histogram,
                        plot_training_data=False,
                        ground_truth_color='darkcyan',
                        prediction_color='tomato')
    self._plot_histogram(x_, y_, nll, None, num_bins=8, ax=self.ax_histogram_twin)

    self.fig_histogram.legend()
    self.fig_histogram.suptitle(
      f'Prediction Epoch {epoch}\nNLL:{logs["val_nll_loss"]:6.3f} | MSE:{logs["val_mse_mean"]:6.3f} ')
    self.fig_histogram.set_tight_layout(True)
    # self.ax_histogram_twin.set_yscale('symlog')

    # save and reset figures
    if self.logdir is not None:
      self.fig_function.savefig(os.path.join(self.graphdir, f'eval_at_epoch_{epoch}.png'))
      self.fig_histogram.savefig(os.path.join(self.graphdir, f'hist_at_epoch_{epoch}.png'))

    self.ax_function.clear()
    self.ax_histogram.clear()
    self.ax_histogram_twin.clear()

  def _plot_function(self,
                     x,
                     pred_mean,
                     pred_std,
                     ax,
                     plot_training_data=True,
                     ground_truth_color='k',
                     prediction_color='red'):
    # plot training data if available
    if self.training_data is not None and plot_training_data:
      x_train, y_train = self.training_data

      if self.input_denormalizer is not None:
        x_train = self.input_denormalizer(X=x_train)

      if self.target_denormalizer is not None:
        y_train = self.target_denormalizer(X=y_train)

      ax.scatter(x_train.ravel(), y_train.ravel(), s=8, marker='+', color='grey', alpha=0.25, label='Training Data')

    # plot true function / variances if available
    if self.true_function is not None:
      _, true_mean, true_variance = [z.ravel() for z in self.true_function(x)]
      ax.plot(x, true_mean, color=ground_truth_color, linestyle=':', label='True Mean')
      ax.fill_between(x,
                      true_mean - 2 * np.sqrt(true_variance),
                      true_mean + 2 * np.sqrt(true_variance),
                      alpha=0.15,
                      color=ground_truth_color)

    # plot prediction
    ax.plot(x, pred_mean, color=prediction_color, label='Predicted Mean')
    ax.fill_between(x, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.15, color=prediction_color)

  def _plot_histogram(self, x, y, nll, nll_un, num_bins, ax):
    # compute bins and indices
    bins = np.linspace(np.min(x) - 1e-2, np.max(x) + 1e-2, num_bins + 1)
    idx_bins = np.digitize(x, bins).ravel()

    _, _, count = np.unique(idx_bins, return_index=True, return_counts=True)
    idx_per_bin = np.split(np.argsort(idx_bins), np.cumsum(count))[:-1]
    bin_means = [np.mean(nll[idx]) for idx in idx_per_bin]
    # bin_means_un = [np.mean(nll_un[idx]) for idx in idx_per_bin]

    # plot step function
    line, = ax.step(bins[:-1], bin_means, where='post', linestyle='dashed', label='NLL')
    ax.plot(bins[-2:], (bin_means[-1], bin_means[-1]), linestyle='dashed', color=line.get_color())

    # line_un, = ax.step(bins[:-1], bin_means_un, where='post', linestyle='dashed', label='N-NLL')
    # ax.plot(bins[-2:], (bin_means_un[-1], bin_means_un[-1]), linestyle='dashed', color=line_un.get_color())

  def on_train_end(self, logs=None):
    order_key = lambda s: int(re.search(r'eval_at_epoch_([0-9]+).png', os.path.basename(s)).group(1))
    image_paths = glob.glob(os.path.join(self.graphdir, 'eval_at_epoch_*.png'))
    image_paths = sorted(image_paths, key=order_key)

    images = [imageio.imread(filename) for filename in image_paths]
    imageio.mimsave(os.path.join(self.graphdir, 'eval.gif'), images, fps=5)

    plt.close(self.fig_function)
    plt.close(self.fig_histogram)


class AdditionalMetricsWithTrueFunctionCallback(tf.keras.callbacks.Callback):

  def __init__(self, validation_data=None, input_denormalizer=None, target_denormalizer=None, true_function=None):
    super(AdditionalMetricsWithTrueFunctionCallback, self).__init__()
    self.validation_data = validation_data
    self.input_denormalizer = input_denormalizer
    self.target_denormalizer = target_denormalizer
    self.true_function = true_function

  def on_epoch_end(self, epoch, logs=None):
    x, y = self.validation_data
    mean, covariance, A = self.model(x)

    # add projection layer if existent
    if hasattr(self.model, 'proj_layer') and hasattr(self.model, 'old_val_means'):
      mean, covariance = self.model.proj_layer(mean, self.model.old_val_means, covariance,
                                               self.model.old_val_covariances)

    y_unnormalized, mean_unnormalized, covariance_unnormalized = y, mean, covariance
    # denormalize input/output for plot
    if self.input_denormalizer is not None:
      x = self.input_denormalizer(X=x)

    if self.target_denormalizer is not None:
      y = self.target_denormalizer(X=y)
      mean, covariance = self.target_denormalizer(X=(mean, covariance))
    
    self._add_unnormalized_metrics_to_log(x, y, mean, covariance, logs)
    # self._add_normalized_metrics_to_log(x, y_unnormalized, mean_unnormalized, covariance_unnormalized, logs)

  def _add_unnormalized_metrics_to_log(self, x, y, pred_mean, pred_covariance, logs):
    if logs is None:
      return

    _, true_mean, true_covariance = self.true_function(x.ravel())

    # cov before is the [B x I x D x D], I = num covs per prediction, assumed 1
    pred_covariance = tf.squeeze(pred_covariance, axis=1)
    true_covariance = tf.cast(true_covariance, dtype=pred_covariance.dtype)

    logs['val_addmetrics/cov_mse'] = np.mean(np.square(true_covariance - pred_covariance))
    logs['val_addmetrics/cov_frob'] = np.mean(utils.frobenius_distance(true_covariance, pred_covariance))
    logs['val_addmetrics/cov_w2'] = np.mean(
      utils.wasserstein_distance(true_covariance, pred_covariance,
                                 tf.eye(tf.shape(pred_covariance)[-1], dtype=pred_covariance.dtype)))
    logs['val_addmetrics/cov_kl'] = np.mean(utils.kl_distance(true_covariance, pred_covariance))

  def _add_normalized_metrics_to_log(self, x, y, pred_mean, pred_covariance, logs):
    if logs is None:
      return
    
    y = y.ravel()
    pred_mean = pred_mean.numpy().ravel()
    pred_std = np.sqrt(pred_covariance.numpy().ravel())

    n_nll = 0.5 * (np.log(2.0 * np.pi) + 2 * np.log(pred_std) + np.square((y - pred_mean) / pred_std))
    
    logs['val_nnll_loss'] = n_nll

class SavePredictionsCallback(tf.keras.callbacks.Callback):

  def __init__(self, training_data=None, validation_data=None, proj_layer=None):
    super(SavePredictionsCallback, self).__init__()
    self.training_data = training_data
    self.validation_data = validation_data
    self.proj_layer = proj_layer

  def on_epoch_begin(self, epoch, logs=None):
    x, _ = self.training_data
    cur_means, cur_covariances, _ = self.model(x, training=False)
    proj_mean, proj_covariance = self.proj_layer(cur_means, self.model.old_means, cur_covariances,
                                                 self.model.old_covariances)

    self.model.old_means.assign(proj_mean)
    self.model.old_covariances.assign(proj_covariance)

  def on_epoch_end(self, epoch, logs=None):
    if self.validation_data is None:
      return

    x, _ = self.validation_data
    cur_means, cur_covariances, _ = self.model(x, training=False)
    proj_mean, proj_covariance = self.proj_layer(cur_means, self.model.old_val_means, cur_covariances,
                                                 self.model.old_val_covariances)

    self.model.old_val_means.assign(proj_mean)
    self.model.old_val_covariances.assign(proj_covariance)


# TODO: add additional figure - KL / W2 / Frob over time for each t
class Graph3DCallback(tf.keras.callbacks.Callback):

  def __init__(self,
               training_data=None,
               validation_data=None,
               input_denormalizer=None,
               target_denormalizer=None,
               true_function=None,
               plot_frequency=100,
               logdir=None):
    super(Graph3DCallback, self).__init__()
    self.training_data = training_data
    self.validation_data = validation_data
    self.plot_frequency = plot_frequency
    self.input_denormalizer = input_denormalizer
    self.target_denormalizer = target_denormalizer
    self.true_function = true_function
    self.logdir = logdir
    self.graphdir = os.path.join(logdir, 'eval_graphs')

    os.makedirs(self.graphdir, exist_ok=True)

  def on_epoch_end(self, epoch, logs=None):
    if self.validation_data is None or epoch % self.plot_frequency != 0:
      return

    x, y = self.validation_data
    mean, covariance, A = self.model(x)

    # add projection layer if existent
    if hasattr(self.model, 'proj_layer') and hasattr(self.model, 'old_val_means'):
      mean, covariance = self.model.proj_layer(mean, self.model.old_val_means, covariance,
                                               self.model.old_val_covariances)

    # denormalize input/output for plot
    if self.input_denormalizer is not None:
      x = self.input_denormalizer(X=x)

    if self.target_denormalizer is not None:
      y = self.target_denormalizer(X=y)
      mean, covariance = self.target_denormalizer(X=(mean, covariance))

    idx = np.argsort(x.ravel())
    x_, pred_mean, pred_covariance = x.ravel()[idx], mean.numpy()[idx].squeeze(), covariance.numpy()[idx].squeeze()

    fig = self._plot(x_, pred_mean, pred_covariance)

    fig.legend()
    fig.suptitle(f'Prediction Epoch {epoch}')
    fig.set_tight_layout(True)

    if self.logdir is not None:
      save_path = os.path.join(self.graphdir, f'eval_at_epoch_{epoch}.png')
      fig.savefig(save_path)
    plt.close(fig)

  def _plot(self, x, pred_mean, pred_covariance):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # # plot training data if available
    # if self.training_data is not None:
    #   x_train, y_train = self.training_data

    #   if self.input_denormalizer is not None:
    #     x_train = self.input_denormalizer(X=x_train)

    #   if self.target_denormalizer is not None:
    #     y_train = self.target_denormalizer(X=y_train)

    #   ax.scatter(x_train.ravel(), y_train.ravel(), s=8, marker='+', color='grey', alpha=0.25, label='Training Data')

    # plot true function / variances if available
    if self.true_function is not None:
      _, true_mean, true_covariance = self.true_function(x)
      ax.plot3D(*true_mean.T, 'k:', label='True Mean')
      for mean_, covariance_ in zip(true_mean, true_covariance):
        plotting.plot_confidence_ellipsoid(mean_[:, np.newaxis], covariance_, ax, color='grey', alpha=0.1)
      # ax.fill_between(x,
      #                 true_mean - 2 * np.sqrt(true_variance),
      #                 true_mean + 2 * np.sqrt(true_variance),
      #                 alpha=0.15,
      #                 color='k')

    # # plot prediction
    ax.plot3D(*pred_mean.T, 'r', label='Predicted Mean')
    for mean_, covariance_ in zip(pred_mean, pred_covariance):
      plotting.plot_confidence_ellipsoid(mean_[:, np.newaxis], covariance_, ax, color='green', alpha=0.1)

    return fig

  def on_train_end(self, logs=None):
    order_key = lambda s: int(re.search(r'eval_at_epoch_([0-9]+).png', os.path.basename(s)).group(1))
    image_paths = glob.glob(os.path.join(self.graphdir, '*.png'))
    image_paths = sorted(image_paths, key=order_key)

    images = [imageio.imread(filename) for filename in image_paths]
    imageio.mimsave(os.path.join(self.graphdir, 'eval.gif'), images, fps=5)


class MemoryUsageCallback(tf.keras.callbacks.Callback):
  '''Monitor memory usage on epoch begin and end.'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))