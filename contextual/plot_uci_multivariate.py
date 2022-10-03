import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
import numpy as np
import pandas as pd


from common import plotting, utils

def create_standard_plot(logdir, function_id, model_ids, key, smoothing=20, ticklabel_fmt='%.2f', adjust_curve=lambda X : X, label_dict=None):
  plot_fnc = plotting.plot_median_confi_curve

  fig, ax = plt.subplots()

  for model_id, color in zip(model_ids, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
    logdir_ = os.path.join(logdir, f'{function_id}/{model_id}')

    curves = utils.get_curves_from_key(logdir_, key)
    for curve in curves:
      curve[:, 1] = adjust_curve(curve[:, 1])
    
    label = label_dict[model_id] if label_dict is not None else model_id
    plot_fnc(curves, ax, plot_singles=False, mean_label=label, color=color, smoothing=smoothing, confidence=0.75)

  # ax.set_yscale('symlog')
  ax.set_xlabel('epochs')
  ax.set_ylabel(key)
  ax.set_title(f'F(x) = {function_id}')

  ax.yaxis.set_minor_locator(tik.AutoMinorLocator())
  ax.yaxis.set_major_formatter(tik.FormatStrFormatter(ticklabel_fmt))

  # fig.legend(loc='upper left', bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
  fig.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes)
  fig.tight_layout()

  return fig, ax


def save_key_plot(logdir, plot_dir, function_id, model_ids, key, ymin, ymax, adjust_curve=lambda X : X, logscale=False, label_dict=None):
  fig, ax = create_standard_plot(logdir, function_id, model_ids, key, adjust_curve=adjust_curve, label_dict=label_dict)
  ax.set_ylim(ymin, ymax)
  if logscale:
    ax.set_yscale('symlog')
  fig.savefig(os.path.join(plot_dir, f'{function_id}_{key.replace("/", "_")}.svg'), dpi=1200)


def print_mean_std_along_keys(logdir, function_id, model_id, main_key='val_losses/unnormalized_nll', side_keys=['val_losses/unnormalized_mse_mean']):
  logdir_ = os.path.join(logdir, f'{function_id}/{model_id}')

  for key in [main_key] + side_keys:
    curves = np.asarray(utils.get_curves_from_key(logdir_, key=main_key))
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
  ### plot experiment 4 : uci multivariate
  ROOT_DIR = os.path.join('logs', 'uci_multivariate')
  
  os.makedirs(plot_dir := os.path.join(ROOT_DIR, 'plots'), exist_ok=True)

  model_ids_plot = ['AdamModel', 'Pitfalls05Model', 'Pitfalls10Model', 'TrustableW2Model']
  model_ids_print = ['AdamModel', 'Pitfalls05Model', 'Pitfalls10Model', 'TrustableW2Model']
  
  if function_id == 'energy':
    save_key_plot(ROOT_DIR, plot_dir, function_id=function_id, model_ids=model_ids_plot, key='val_losses/unnormalized_mse_mean', ymin=0.0, ymax=100.0)
    save_key_plot(ROOT_DIR, plot_dir, function_id=function_id, model_ids=model_ids_plot, key='val_losses/unnormalized_nll', ymin=0.0, ymax=50.0)  
    
    for model_id in model_ids_print:
      print_mean_std_along_keys(ROOT_DIR, function_id, model_id, side_keys=['val_losses/unnormalized_mse_mean'])
  elif function_id == 'carbon':
    save_key_plot(ROOT_DIR, plot_dir, function_id=function_id, model_ids=model_ids_plot, key='val_losses/unnormalized_mse_mean', ymin=0.0, ymax=0.01)
    save_key_plot(ROOT_DIR, plot_dir, function_id=function_id, model_ids=model_ids_plot, key='val_losses/unnormalized_nll', ymin=-15.0, ymax=60.0)  
    
    for model_id in model_ids_print:
      print_mean_std_along_keys(ROOT_DIR, function_id, model_id, side_keys=['val_losses/unnormalized_mse_mean'])
      
  plt.show()
  
  quit()