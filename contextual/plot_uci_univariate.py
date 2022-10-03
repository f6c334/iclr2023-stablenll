import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
import numpy as np
import pandas as pd


from common import plotting, utils


def create_standard_plot(logdir, function_id, model_ids, key, smoothing=10, ticklabel_fmt='%.2f'):
  plot_fnc = plotting.plot_median_confi_curve

  fig, ax = plt.subplots(figsize=(16, 9))

  for model_id, color in zip(model_ids, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
    logdir_ = os.path.join(logdir, f'{function_id}/{model_id}')

    curves = utils.get_curves_from_key(logdir_, key)
    plot_fnc(curves, ax, plot_singles=False, mean_label=model_id, color=color, smoothing=smoothing, confidence=0.75)

  # ax.set_yscale('symlog')
  ax.set_xlabel('epochs')
  ax.set_ylabel(key)
  ax.set_title(f'F(x) = {function_id}')

  ax.yaxis.set_minor_locator(tik.AutoMinorLocator())
  ax.yaxis.set_major_formatter(tik.FormatStrFormatter(ticklabel_fmt))

  fig.legend()
  fig.tight_layout()

  return fig, ax


def save_key_plot(logdir, plot_dir, function_id, model_ids, key, ymin, ymax, logscale=False):
  fig, ax= create_standard_plot(logdir, function_id, model_ids, key)
  ax.set_ylim(ymin, ymax)
  if logscale:
    ax.set_yscale('symlog')
  fig.savefig(os.path.join(plot_dir, f'{function_id}_{key.replace("/", "_")}.svg'), dpi=1200)


def print_mean_std_along_keys(logdir, function_id, model_id, main_key='val_losses/unnormalized_nll', side_keys=['val_losses/unnormalized_mse_mean']):
  logdir_ = os.path.join(logdir, f'{function_id}/{model_id}')

  for key in [main_key] + side_keys:
    curves = np.asarray(utils.get_curves_from_key(logdir_, key=key))
    mean_curve = np.mean(curves[:, :, 1], axis=0)
    std_curve = np.std(curves[:, :, 1], axis=0)
    
    min_idx = mean_curve.argmin()
    
    print(model_id, key, f'${mean_curve[min_idx]:e} \\pm {std_curve[min_idx]:e}$')


if __name__ == '__main__':
  import sys
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  plt.style.use(['./science.mplstyle'])
  # mpl.rcParams['legend.fontsize'] = 'small'
  mpl.rcParams['figure.figsize'] = [5, 3.5]
  
  
  function_id = sys.argv[1]
  ###########################################################
  ### plot experiment 3 : uci univariate
  ROOT_DIR = os.path.join('logs', 'uci_univariate')
  
  os.makedirs(plot_dir := os.path.join(ROOT_DIR, 'plots'), exist_ok=True)

  model_ids = [ 'AdamModel', 'Pitfalls05Model', 'Pitfalls10Model', 'TrustableW2Model']

  for model_id in model_ids:
    print_mean_std_along_keys(ROOT_DIR, function_id, model_id, main_key='val_losses/unnormalized_nll', side_keys=['val_losses/unnormalized_mse_mean'])

  quit()