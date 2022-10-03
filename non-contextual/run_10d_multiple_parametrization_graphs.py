import itertools
import multiprocessing
import os
import time

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import scipy as sc
import tensorflow as tf
import tqdm as tq

from sgd import sgd_gaussian_optimization
from adam import adam_gaussian_optimization
from natural import natural_gaussian_optimization
from pitfalls import pitfalls_gaussian_optimization
from trustregions import trp_gaussian_optimization
from tractable import tractable_gaussian_optimization
from gaussnewton import gaussnewton_gaussian_optimization
from traptable import traptable_gaussian_optimization

from parametrizations import VanillaParametrization, CholeskyParametrization, SqrtCovarianceParametrization
from parametrizations import TractableCholeskyParametrization, ApproximateTractableCholeskyParametrization

from utils.plotting import  plot_mean_std_curve, plot_nd_gaussian_kl
from utils.common import seeding, rotation_matrix_around_standard_spans, kl_mvn

import tr_projections.tensorflow.trust_region_layers as trust_region_layers


### HYPERPARAMS ######
SEED = 0
seeding(SEED, tf_deterministic=False)

SAMPLES_PER_CASE = 10

dims = 10
iterations, batch_size, mini_batches = 400, 128, 8
alpha = 5e-2

folder = './non-contextual/figures/10d_graphs_5e-2'


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
  [{  'initial_mean' : random_mean(low=-5, high=5, size=(dims)),    'initial_covariance' : random_covariance(optimal_eigval_dist(size=(dims))),
      'true_mean' : random_mean(low=-5, high=5, size=(dims)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(dims)))     }
    for _ in range(SAMPLES_PER_CASE)],
  # 2. small - optimal      : expect explosion at the beginning
  [{  'initial_mean' : random_mean(low=-5, high=5, size=(dims)),    'initial_covariance' : random_covariance(small_eigval_dist(size=(dims))),
      'true_mean' : random_mean(low=-5, high=5, size=(dims)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(dims)))     }
    for _ in range(SAMPLES_PER_CASE)],
  # 3. large - optimal      : expect very slow convergence
  [{  'initial_mean' : random_mean(low=-5, high=5, size=(dims)),    'initial_covariance' : random_covariance(large_eigval_dist(size=(dims))),
      'true_mean' : random_mean(low=-5, high=5, size=(dims)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(dims)))     }
    for _ in range(SAMPLES_PER_CASE)],
  # 4. inoptimal - optimal  : expect jumps at beginning, then slow convergence
  [{  'initial_mean' : random_mean(low=-5, high=5, size=(dims)),    'initial_covariance' : random_covariance(small_eigval_dist(size=(dims))),
      'true_mean' : random_mean(low=-5, high=5, size=(dims)),       'true_covariance' : random_covariance(optimal_eigval_dist(size=(dims)))     }
    for _ in range(SAMPLES_PER_CASE)],
]
######################

## comparison between all cases
parametrizations = [VanillaParametrization, CholeskyParametrization, SqrtCovarianceParametrization]
algorithms = [
  ('sgd',         sgd_gaussian_optimization,          {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('adam',        adam_gaussian_optimization,         {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('natural',     natural_gaussian_optimization,      {'delta' : 0.1, 'iterations' : 50, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('pitfalls05',  pitfalls_gaussian_optimization,     {'alpha' : alpha, 'beta' : 0.5, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  #('pitfalls10',  pitfalls_gaussian_optimization,     {'alpha' : alpha, 'beta' : 1.0, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpFrob',     trp_gaussian_optimization,          {'alpha' : alpha, 'proj_layer' : trust_region_layers.FrobProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpW2',       trp_gaussian_optimization,          {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpKL',       trp_gaussian_optimization,          {'alpha' : alpha, 'proj_layer' : trust_region_layers.KLProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('gaussnewton', gaussnewton_gaussian_optimization,  {'alpha' : alpha, 'iterations' : 50, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpW2Force',  trp_gaussian_optimization,          {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ForceProjectionLayer(mean_bound=0.1, covariance_bound=0.1), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
]

parametrizations_ = [TractableCholeskyParametrization, ApproximateTractableCholeskyParametrization]
algorithms_ = [
  ('tractable',   tractable_gaussian_optimization,    {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpTracFrob', traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.FrobProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpTracW2',   traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('trpTracKL',   traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.KLProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
]


# comparison for tractable / trpW2 / ...
parametrizations__ = [ApproximateTractableCholeskyParametrization]
algorithms__ = [
  ('tractable',   tractable_gaussian_optimization,    {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('01trpTracW2',   traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=0.1, covariance_bound=0.1), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('10trpTracW2',   traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=1.0, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  ('100trpTracW2',   traptable_gaussian_optimization,    {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=10.0, covariance_bound=10.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
]



###############################################################################
### DATA FUNCTIONS ############################################################
get_run_instance_name = lambda path, algo_name, param_name, case_id, instance_id : \
  os.path.join(path, f'{algo_name}_{param_name}_{case_id}_{instance_id}.npy')

def load_run_instance(path, algo_name, param_name, case_id, instance_id):
  save_name = get_run_instance_name(path, algo_name, param_name, case_id, instance_id)
  log = np.load(save_name, allow_pickle=True)

  times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9   # in seconds
  means = np.asarray([log_['mean'].squeeze() for log_ in log])
  covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])

  return times, means, covariances
###############################################################################
###############################################################################


###############################################################################
### RUN ALGORITHMS IF NEEDED ##################################################
def init_worker():
  import signal
  import sys

  signal.signal(signal.SIGINT, signal.SIG_IGN)
  sys.stdout = None

def worker_run(algo_name, algo_fnc, algo_params, parametrization, case_id, instance_id, instance_config, path):
  import tensorflow as tf

  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  seeding(SEED, tf_deterministic=False)

  # execute and save if necessary
  save_name = get_run_instance_name(path, algo_name, parametrization.__name__, case_id, instance_id)
  if not os.path.exists(save_name):
    with tf.device('/gpu:0'):
      log = algo_fnc(**instance_config, phi=parametrization(), **algo_params)
    np.save(save_name, log)


def run_algorithms(algorithms, parametrizations, case_configs, path, pool_size=3):
  os.makedirs(path, exist_ok=True)

  # generate pool and execute, wait till finished or input is detected
  worker_instances_params = []

  # two fors for algorithm specifics
  for algo in algorithms:
    for parametrization in parametrizations:
      
      # two fors for the optimization problem values
      for case_id, case_config in enumerate(case_configs):
        for instance_id, instance_config in enumerate(case_config):
          worker_instances_params.append((*algo, parametrization, case_id, instance_id, instance_config, path))
  
  # run processes and status bar
  pbar = tq.tqdm(total=len(worker_instances_params))
  update = lambda *a : pbar.update()
  
  pool = multiprocessing.Pool(pool_size, init_worker)
  jobs = [pool.starmap_async(worker_run, [instance_params], callback=update) for instance_params in worker_instances_params]

  try:
    while True:
      time.sleep(5.0)

      if all([job.ready() for job in jobs]):
        break
  except KeyboardInterrupt:
    pool.terminate()
    pool.join()
    quit() # keyboard interrupt quits program
  finally:
    pool.close()
    pool.join()
###############################################################################
###############################################################################


### KL DIVERGENCE PLOTS #######################################################
###############################################################################
def plot_kl_groupby_parametrization(algorithms, parametrizations, case_configs, data_path, path, normalize_kl=False, on_time=False, log_scale=True):
  os.makedirs(path, exist_ok=True)
  
  matplotlib.use('Agg')

  for algo_name, _, _ in algorithms:
    for case_id, case_config in enumerate(case_configs):
      for instance_id, instance_config in enumerate(case_config):
        _, _, true_mean, true_covariance = instance_config.values()

        fig, ax = plt.subplots(1, 1)

        for parametrization in parametrizations:
          times, means, covariances = load_run_instance(data_path, algo_name, parametrization.__name__, case_id, instance_id)
          
          plot_nd_gaussian_kl(means, covariances,
                              true_mean, true_covariance,
                              label=parametrization.__name__.removesuffix('Parametrization'),
                              ax=ax,
                              times=times if on_time else None,
                              normalize_kl=normalize_kl,
                              log_scale=True)

        if not on_time:
          ax2 = ax.twiny()
          ax2.set_xticks([x for x in ax.get_xticks() if x < times.size and x >= 0])
          ax2.set_xbound((0, times.size))
          ax2.set_xticklabels([round(times[int(x)], 2) for x in ax2.get_xticks()])
          ax2.set_xlabel('t [s]')

        ax.legend()
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        
        # fig.suptitle(f'{algo_name.upper()}, Case #{case_id}')
        fig.set_tight_layout(True)


        file_suffix = 'kl_on_time' if on_time else 'kl_on_iter'
        file_suffix += 'n' if normalize_kl else ''
        fig.savefig(os.path.join(path, f'{algo_name}_{case_id}_{instance_id}_{file_suffix}.svg'), dpi=1200)
        plt.close(fig=fig)
###############################################################################
def plot_kl_groupby_parametrization_cases(algorithms, parametrizations, case_configs, data_path, path, normalize_kl=False, on_time=False, log_scale=True):
  os.makedirs(path, exist_ok=True)
  
  matplotlib.use('Agg')

  # two fors for algorithm specifics
  for algo_name, algo_fnc, algo_params in algorithms:

    # two fors for the optimization problem values
    for case_id, case_config in enumerate(case_configs):
      fig, ax = plt.subplots(1, 1)
      
      for parametrization in parametrizations:
        curves = []

        for instance_id, instance_config in enumerate(case_config):
          initial_mean, initial_covariance, true_mean, true_covariance = instance_config.values()

          times, means, covariances = load_run_instance(data_path, algo_name, parametrization.__name__, case_id, instance_id)
          
          kl = np.asarray([kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)])
          metric = 1 - np.exp(- kl) if normalize_kl else kl

          if on_time:
            curves.append(np.asarray(list(zip(times, metric))))
          else:
            curves.append(np.asarray(list(enumerate(metric))))
        
        plot_mean_std_curve(curves, ax, plot_singles=False, mean_label=f'{parametrization.__name__}', color=next(ax._get_lines.prop_cycler)['color'])

      if log_scale:
        ax.set_yscale('log')
      
      ax.set_xlabel('t [s]' if on_time else 'Iterations')
      ax.set_ylabel('NKL' if normalize_kl else 'KL')

      ax.legend()
      ax.ticklabel_format(axis='x', style='plain', useOffset=False)

      # fig.suptitle(f'{algo_name.upper()}, Case #{case_id}')
      fig.set_tight_layout(True)
      

      file_suffix = 'kl_on_time' if on_time else 'kl_on_iter'
      file_suffix += 'n' if normalize_kl else ''
      fig.savefig(os.path.join(path, f'{algo_name}_{case_id}_{file_suffix}.svg'), dpi=1200)
      plt.close(fig=fig)
###############################################################################
def plot_kl_groupby_cases_algos(algorithms, parametrizations, case_configs, data_path, path, normalize_kl=False, on_time=False, log_scale=True):
  os.makedirs(path, exist_ok=True)
  
  matplotlib.use('Agg')

  # two fors for the optimization problem values
  for case_id, case_config in enumerate(case_configs):
    for parametrization in parametrizations:
      fig, ax = plt.subplots(1, 1)
      
      # two fors for algorithm specifics
      for algo_name, algo_fnc, algo_params in algorithms:
        curves = []

        for instance_id, instance_config in enumerate(case_config):
          initial_mean, initial_covariance, true_mean, true_covariance = instance_config.values()

          times, means, covariances = load_run_instance(data_path, algo_name, parametrization.__name__, case_id, instance_id)
          
          kl = np.asarray([kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)])
          metric = 1 - np.exp(- kl) if normalize_kl else kl

          if on_time:
            curves.append(np.asarray(list(zip(times, metric))))
          else:
            curves.append(np.asarray(list(enumerate(metric))))
        
        plot_mean_std_curve(curves, ax, plot_singles=False, mean_label=f'{algo_name}', color=next(ax._get_lines.prop_cycler)['color'])

      if log_scale:
        ax.set_yscale('log')
      
      ax.set_xlabel('t [s]' if on_time else 'Iterations')
      ax.set_ylabel('NKL' if normalize_kl else 'KL')

      ax.legend()
      ax.ticklabel_format(axis='x', style='plain', useOffset=False)

      # fig.suptitle(f'{algo_name.upper()}, Case #{case_id}')
      fig.set_tight_layout(True)
      

      file_suffix = 'kl_on_time' if on_time else 'kl_on_iter'
      file_suffix += 'n' if normalize_kl else ''
      fig.savefig(os.path.join(path, f'{parametrization.__name__}_{case_id}_{file_suffix}.svg'), dpi=1200)
      plt.close(fig=fig)
###############################################################################
def plot_kl_groupby_specific_algos(algorithms, case_configs, data_path, path, normalize_kl=False, on_time=False, log_scale=True, name_map=None):
  os.makedirs(path, exist_ok=True)
  
  matplotlib.use('Agg')

  # two fors for the optimization problem values
  for case_id, case_config in enumerate(case_configs):
    # for parametrization in parametrizations:
    fig, ax = plt.subplots(1, 1)
    
    # two fors for algorithm specifics
    for algo_name, algo_fnc, algo_params in algorithms:      
      parametrization = CholeskyParametrization
      if algo_name in ['tractable', 'trpTracFrob', 'trpTracW2', 'trpTracKL']:
        parametrization = ApproximateTractableCholeskyParametrization
      
      curves = []

      for instance_id, instance_config in enumerate(case_config):
        initial_mean, initial_covariance, true_mean, true_covariance = instance_config.values()

        times, means, covariances = load_run_instance(data_path, algo_name, parametrization.__name__, case_id, instance_id)
        
        kl = np.asarray([kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)])
        metric = 1 - np.exp(- kl) if normalize_kl else kl

        if on_time:
          curves.append(np.asarray(list(zip(times, metric))))
        else:
          curves.append(np.asarray(list(enumerate(metric))))
      
      a_name = algo_name if name_map is None else name_map[algo_name]
      plot_mean_std_curve(curves, ax, plot_singles=False, mean_label=f'{a_name}', color=next(ax._get_lines.prop_cycler)['color'])

    if log_scale:
      ax.set_yscale('log')
    
    ax.set_xlabel('t [s]' if on_time else 'Iterations')
    ax.set_ylabel('NKL' if normalize_kl else 'KL')

    ax.legend(ncol=2, loc='upper right')
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

    # fig.suptitle(f'{algo_name.upper()}, Case #{case_id}')
    fig.set_tight_layout(True)
    

    file_suffix = 'kl_on_time' if on_time else 'kl_on_iter'
    file_suffix += 'n' if normalize_kl else ''
    fig.savefig(os.path.join(path, f'{parametrization.__name__}_{case_id}_{file_suffix}.svg'), dpi=1200)
    plt.close(fig=fig)
###############################################################################
###############################################################################
###############################################################################
def compute_metrics_on_data(algorithms, parametrizations, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1):
  aucs = np.empty(shape=(len(algorithms), len(parametrizations), len(case_configs), len(case_configs[0])))
  conv_it = np.empty(shape=(len(algorithms), len(parametrizations), len(case_configs), len(case_configs[0])))
  conv_t = np.empty(shape=(len(algorithms), len(parametrizations), len(case_configs), len(case_configs[0])))

  for algo_id, (algo_name, algo_fnc, algo_params) in enumerate(algorithms):
    for case_id, case_config in enumerate(case_configs):
      for parametrization_id, parametrization in enumerate(parametrizations):
        for instance_id, instance_config in enumerate(case_config):
          _, _, true_mean, true_covariance = instance_config.values()

          times, means, covariances = load_run_instance(data_path, algo_name, parametrization.__name__, case_id, instance_id)
          
          kl = np.asarray([kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)])
          metric = 1 - np.exp(- kl) if normalize_kl else kl
          
          conv_idx = (kl < conv_threshold).nonzero()[0]
          conv_idx = -1 if len(conv_idx) < 1 else conv_idx[0]

          aucs[algo_id, parametrization_id, case_id, instance_id] = np.mean(metric)
          conv_it[algo_id, parametrization_id, case_id, instance_id] = conv_idx
          conv_t[algo_id, parametrization_id, case_id, instance_id] = times[conv_idx]
  return aucs, conv_it, conv_t

def plot_grouped_boxplot(data, ax, offset_position=0, bar_width=0.1, group_width=0.66, bar_labels=None, group_labels=None, cmap_name='jet'):
  num_groups, num_bars_per_group, _ = data.shape
  
  cmap = cm.get_cmap(cmap_name)
  colors = [cmap(i / num_bars_per_group) for i in range(num_bars_per_group)]
  
  start_idx = [group_id + offset_position for group_id in range(num_groups)]
  group_mids = [start_idx[group_id] + group_width / 2 for group_id in range(num_groups)]

  for group_id in range(num_groups):
    bars_data = data[group_id].T
    
    left_border, right_border = group_mids[group_id] - group_width / 2, group_mids[group_id] + group_width / 2
    positions = np.linspace(left_border, right_border, num_bars_per_group + 2)[1:-1]
    bplot = ax.boxplot(bars_data, positions=positions, 
                      widths=bar_width, 
                      patch_artist=True, 
                      manage_ticks=False,
                      flierprops={'marker': 'o', 'markersize': 2})
    
    for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)


  if group_labels is not None:
    new_ticks = list(ax.get_xticks()) + group_mids
    new_ticklabels = list(ax.get_xticklabels()) + group_labels

    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_ticklabels, rotation=45, ha='right')

  if bar_labels is not None:
    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, bar_labels)]
    ax.legend(handles=handles, loc='upper right')

  

def plot_auc_graphs(algorithms, parametrizations, case_configs, data_path, path):
  os.makedirs(path, exist_ok=True)

  save_name = os.path.join(data_path, f'{algorithms[0][0]}_{algorithms[-1][0]}_aucs.npy')
  if not os.path.exists(save_name):
    aucs, _, _ = compute_metrics_on_data(algorithms, parametrizations, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1)
    np.save(save_name, aucs)

  aucs = np.load(save_name, allow_pickle=True)
  
  ## COMPLETE PLOT WITH ALL ALGOS, PARAMS, CASES (grouped by params)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(16, 9)

  ax.set_xticks([])

  num_groups = aucs.shape[1]
  for algo_id, (algo_name, _, _) in enumerate(algorithms):
    bar_labels = [f'Case #{i}' for i in range(len(case_configs))]
    group_labels = [f'{algo_name} ({param.__name__.removesuffix("Parametrization")})' for param in parametrizations]
    
    plot_grouped_boxplot(aucs[algo_id], ax, 
                        offset_position=num_groups * algo_id, 
                        bar_labels=bar_labels, 
                        group_labels=group_labels)
    
    if algo_id > 0:
      ax.axvline(num_groups * algo_id - 0.15, linestyle='--', color='black', alpha=0.15)
  
  ax.set_yscale('log')
  ax.set_ylabel('$AUC_{KL}$ confidence')

  # fig.suptitle(f'Area under Curve Confidence\nfor 10 runs per algorithm/case')
  fig.set_tight_layout(True)
  
  fig.savefig(os.path.join(path, f'auc_boxplot_complete_gbparam-{algorithms[0][0]}_{algorithms[-1][0]}.svg'), dpi=1200)


  ## COMPLETE PLOT WITH ALL ALGOS, PARAMS, CASES (grouped by cases)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(16, 9)

  ax.set_xticks([])
  
  num_groups = aucs.shape[2]
  for algo_id, (algo_name, _, _) in enumerate(algorithms):
    bar_labels = [f'{param.__name__.removesuffix("Parametrization")}' for param in parametrizations]
    group_labels = [f'{algo_name} (Case #{i})' for i in range(len(case_configs))]
    
    plot_grouped_boxplot(np.transpose(aucs, (0, 2, 1, 3))[algo_id], ax, 
                        offset_position=num_groups * algo_id,
                        bar_labels=bar_labels, 
                        group_labels=group_labels)
    
    if algo_id > 0:
      ax.axvline(num_groups * algo_id - 0.15, linestyle='--', color='black', alpha=0.15)
  
  ax.set_yscale('log')
  ax.set_ylabel('$AUC_{KL}$ confidence')

  # fig.suptitle(f'Area under Curve Confidence\nfor 10 runs per algorithm/case')
  fig.set_tight_layout(True)
  
  fig.savefig(os.path.join(path, f'auc_boxplot_complete_gbcase-{algorithms[0][0]}_{algorithms[-1][0]}.svg'), dpi=1200)


  ## ONE PLOT PER CASE WITH ALL ALGOS, PARAMS (grouped by params)
  for case_id, _ in enumerate(case_configs):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 9)

    ax.set_xticks([])

    bar_labels = [f'{param.__name__.removesuffix("Parametrization")}' for param in parametrizations]
    group_labels = [f'{algo_name}' for algo_name, _, _ in algorithms]

    data__ = aucs[:,:,case_id,:]
    plot_grouped_boxplot(data__, ax,
                        offset_position=0,
                        bar_labels=bar_labels,
                        group_labels=group_labels)
    
  
    ax.set_yscale('log')
    ax.set_ylabel('$AUC_{KL}$ confidence')

    # fig.suptitle(f'Area under Curve Confidence\nfor 10 runs per algorithm\nCase #{case_id}')
    fig.set_tight_layout(True)
  
    fig.savefig(os.path.join(path, f'auc_boxplot_c{case_id}-{algorithms[0][0]}_{algorithms[-1][0]}.svg'), dpi=1200)


def plot_mixed_auc_graphs(algorithms, parametrizations, algorithms_, parametrizations_, case_configs, data_path, path):
  os.makedirs(path, exist_ok=True)

  save_name = os.path.join(data_path, f'{algorithms[0][0]}_{algorithms[-1][0]}_aucs.npy')
  if not os.path.exists(save_name):
    aucs, _, _ = compute_metrics_on_data(algorithms, parametrizations, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1)
    np.save(save_name, aucs)

  save_name_ = os.path.join(data_path, f'{algorithms_[0][0]}_{algorithms_[-1][0]}_aucs.npy')
  if not os.path.exists(save_name_):
    aucs_, _, _ = compute_metrics_on_data(algorithms_, parametrizations_, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1)
    np.save(save_name_, aucs_)

  aucs = np.load(save_name, allow_pickle=True)
  aucs_ = np.load(save_name_, allow_pickle=True)
  
  ## COMPLETE PLOT WITH ALL ALGOS, PARAMS, CASES (grouped by params)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(16, 9)

  ax.set_xticks([])

  num_groups = aucs.shape[1]
  for algo_id, (algo_name, _, _) in enumerate(algorithms):
    bar_labels = [f'Case #{i}' for i in range(len(case_configs))]
    group_labels = [f'{algo_name} ({param.__name__.removesuffix("Parametrization")})' for param in parametrizations]
    
    plot_grouped_boxplot(aucs[algo_id], ax, 
                        offset_position=num_groups * algo_id, 
                        bar_labels=bar_labels, 
                        group_labels=group_labels)
    
    if algo_id > 0:
      ax.axvline(num_groups * algo_id - 0.15, linestyle='--', color='black', alpha=0.15)
  
  for algo_id, (algo_name, _, _) in enumerate(algorithms_):
    bar_labels = [f'Case #{i}' for i in range(len(case_configs))]
    group_labels = [f'{algo_name} ({param.__name__.removesuffix("Parametrization")})' for param in parametrizations_]
    
    plot_grouped_boxplot(aucs_[algo_id], ax, 
                        offset_position=num_groups * (algo_id + len(algorithms)), 
                        bar_labels=bar_labels, 
                        group_labels=group_labels)
    
    ax.axvline(num_groups * (algo_id + len(algorithms)) - 0.15, linestyle='--', color='black', alpha=0.15)
  
  ax.set_yscale('log')
  ax.set_ylabel('$AUC_{KL}$ confidence')

  # fig.suptitle(f'Area under Curve Confidence\nfor 10 runs per algorithm/case')
  fig.set_tight_layout(True)
  
  fig.savefig(os.path.join(path, f'auc_boxplot_combined_gbparam-{algorithms[0][0]}_{algorithms_[-1][0]}.svg'), dpi=1200)




  



def plot_auc_tables(algorithms, parametrizations, case_configs, data_path, path):
  os.makedirs(path, exist_ok=True)

  save_name = os.path.join(data_path, f'{algorithms[0][0]}_{algorithms[-1][0]}_aucs_.npz')
  if not os.path.exists(save_name):
    aucs, conv_it, conv_t= compute_metrics_on_data(algorithms, parametrizations, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1)
    np.savez(save_name, aucs, conv_it, conv_t)

  aucs, conv_it, conv_t = np.load(save_name, allow_pickle=True).values()

  
  # aucs = [algos, params, cases, normals]
  mean_aucs = np.mean(aucs, axis=3) # [algos, params, cases]
  mean_aucs = np.reshape(mean_aucs, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  mean_conv_it = np.mean(conv_it, axis=3) # [algos, params, cases]
  mean_conv_it = np.reshape(mean_conv_it, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  mean_conv_t = np.mean(conv_t, axis=3) # [algos, params, cases]
  mean_conv_t = np.reshape(mean_conv_t, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  fig = plt.figure(figsize=(20, 1.5 * len(algorithms)))
  ax = fig.add_subplot(111)

  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  plt.box(on=None)
  
  """
  header_0 = ax.table(cellText=[[''] for i in range(len(algorithms))],
                     rowLabels=[f'{algorithms[i][0]}' for i in range(len(algorithms))],
                     loc='center',
                     bbox=[0.0, 0.0, 0.1, 1.0])
  
  header_1 = ax.table(cellText=[['']],
                     colLabels=['Just Hail'],
                     loc='center',
                     bbox=[0.8, -0.1, 0.2, 0.1])
  
  the_table = plt.table(cellText=cell_text,
                        rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      bbox=[0, -0.35, 1.0, 0.3])
  """

  cell_text = []
  for i in range(mean_aucs.shape[0]):
    mean_auc_str = ['---' if np.isnan(mean_aucs[i][j]) else f'{mean_aucs[i][j]:2.2E}' for j in range(mean_aucs.shape[1])]
    mean_conv_it_str = ['---' if mean_conv_it[i][j] < 0 else f'{mean_conv_it[i][j]:.1f}it' for j in range(mean_aucs.shape[1])]
    mean_conv_t_str = [f'{mean_conv_t[i][j]:.1f}s' for j in range(mean_aucs.shape[1])]
    text = [f'{mauc_str} | {mcit_str} | {mct_str}' for mauc_str, mcit_str, mct_str in zip(mean_auc_str, mean_conv_it_str, mean_conv_t_str)]
    cell_text.append(text)

  #cell_text = np.around(mean_aucs, decimals=2)
  inner_column_labels = [f'Case {i}' for i in range(len(case_configs))]
  inner_row_labels = [f'{algorithms[i][0]}\n{parametrizations[j].__name__.removesuffix("Parametrization")}' for i in range(len(algorithms)) for j in range(len(parametrizations))]
  table = plt.table(cellText=cell_text,
                        rowLabels=inner_row_labels,
                        #rowColours=colors,
                        colLabels=inner_column_labels,
                        #colWidths=[1.5 for x in range(len(case_configs))],
                        #rowWidths=[0.5 for x in range(len(case_configs) + 1)],
                        loc='center',
                        bbox=[0.0, 0.0, 1.0, 1.0]
                        )
  for key, cell in table.get_celld().items():
    cell.set_linewidth(1)
  #table.auto_set_font_size(False)
  #table.set_fontsize(25)
  #table.scale(1.5, 1.5)  # may help
  
  fig.set_tight_layout(True)
  fig.savefig(os.path.join(path, f'table_{algorithms[0][0]}_{algorithms[-1][0]}.svg'), dpi=1200)
  #plt.show()




def save_latex_table(algorithms, parametrizations, case_configs, data_path, path):
  os.makedirs(path, exist_ok=True)

  save_name = os.path.join(data_path, f'{algorithms[0][0]}_{algorithms[-1][0]}_aucs_.npz')
  if not os.path.exists(save_name):
    aucs, conv_it, conv_t= compute_metrics_on_data(algorithms, parametrizations, case_configs, data_path, normalize_kl=False, conv_threshold=1e-1)
    np.savez(save_name, aucs, conv_it, conv_t)

  aucs, conv_it, conv_t = np.load(save_name, allow_pickle=True).values()

  # aucs = [algos, params, cases, normals]
  mean_aucs = np.mean(aucs, axis=3) # [algos, params, cases]
  mean_aucs = np.reshape(mean_aucs, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  mean_conv_it = np.mean(conv_it, axis=3) # [algos, params, cases]
  mean_conv_it = np.reshape(mean_conv_it, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  mean_conv_t = np.mean(conv_t, axis=3) # [algos, params, cases]
  mean_conv_t = np.reshape(mean_conv_t, newshape=(len(algorithms) * len(parametrizations), len(case_configs)))

  import pandas as pd

  inner_row_labels = [f'{algorithms[i][0]}\n{parametrizations[j].__name__.removesuffix("Parametrization")}' for i in range(len(algorithms)) for j in range(len(parametrizations))]
  inner_column_labels = [[f'Case {i} - {m}' for m in ['M-AUC', 'ITER', 'TIME']] for i in range(len(case_configs))]
  inner_column_labels = [item for sublist in inner_column_labels for item in sublist]

  concatenated_metrics = [np.stack([mean_aucs[:, i], mean_conv_it[:, i], mean_conv_t[:, i]], axis=1) for i in range(len(case_configs))]
  rows = np.concatenate(concatenated_metrics, axis=1)

  df = pd.DataFrame(rows, index=inner_row_labels, columns=inner_column_labels)
  df.to_latex(os.path.join(path, f'{algorithms[0][0]}_{algorithms[-1][0]}_table.tex'))








def plot_distribution(path):

  fig = plt.figure(figsize=(16, 9))
  ax = fig.add_subplot(111, label="1")
  ax2 = fig.add_subplot(111, label="2", frame_on=False)
  
  # plot bottom, left axis
  x = np.linspace(0.5, 1.5, 100)
  probs = sc.stats.beta.pdf(x, 2, 2, loc=0.5)
  ax.plot(x, probs, label='optimal', color='blue')
  ax.fill_between(x, probs, interpolate=True, alpha=0.1)
  ax.text(1, 1, '#0', fontsize=20, color='blue', ha='center', va='center')

  x = np.linspace(0.01, 1.01, 100)
  probs = sc.stats.beta.pdf(x, 0.5, 8, loc=0.01)
  ax.plot(x, probs, label='small', color='orange')
  ax.fill_between(x, probs, interpolate=True, alpha=0.1)
  ax.text(0.15, 0.5, '#1', fontsize=20, color='orange', ha='center', va='center')

  # plot top, right axis
  x = np.linspace(0.0, 1.0, 10000)
  probs = sc.stats.beta.pdf(x, 8, 0.5)
  ax2.plot(x * 100, probs, label='large', color='green', linestyle='--')
  ax2.fill_between(x * 100, probs, interpolate=True, color='green', alpha=0.1)
  ax2.text(85, 0.5, '#2', fontsize=20, color='green', ha='center', va='center')

  ax.set_ylim(ymin=0.0, ymax=3.0)
  ax.set_xlabel('$\lambda$, solid line')
  ax.set_ylabel('$p(\lambda)$, solid line')

  ax2.set_ylim(ymin=0.0, ymax=1.0)
  ax2.xaxis.tick_top()
  ax2.yaxis.tick_right()
  ax2.set_xlabel('$\lambda$, dotted line')
  ax2.set_ylabel('$p(\lambda)$, dotted line')
  ax2.xaxis.set_label_position('top')
  ax2.yaxis.set_label_position('right')

  # fig.suptitle('Eigenvalue Distributions per ND-Case')
  fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

  fig.savefig(os.path.join(path, f'distributions_visualization.svg'), dpi=1200)


if __name__ == "__main__":
  # nvidia-smi -l
  
  import matplotlib as mpl
  plt.style.use(['./science.mplstyle'])
  mpl.rcParams['figure.figsize'] = [6.2, 6.2]


  data_folder = os.path.join(folder, 'data')
  run_algorithms(algorithms, parametrizations, case_configs, data_folder, pool_size=1)
  run_algorithms(algorithms_, parametrizations_, case_configs, data_folder, pool_size=1)


  kl_groupby_parametrization_on_iter_folder = os.path.join(folder, 'kl_on_iter')
  plot_kl_groupby_parametrization(algorithms, parametrizations, case_configs, data_folder, kl_groupby_parametrization_on_iter_folder)
  plot_kl_groupby_parametrization(algorithms_, parametrizations_, case_configs, data_folder, kl_groupby_parametrization_on_iter_folder)

  kl_groupby_parametrization_on_time_folder = os.path.join(folder, 'kl_on_time')
  plot_kl_groupby_parametrization(algorithms, parametrizations, case_configs, data_folder, kl_groupby_parametrization_on_time_folder, on_time=True)
  plot_kl_groupby_parametrization(algorithms_, parametrizations_, case_configs, data_folder, kl_groupby_parametrization_on_time_folder, on_time=True)


  kl_groupby_parametrization_cases_on_iter_folder = os.path.join(folder, 'kl_on_iter_meanstd')
  plot_kl_groupby_parametrization_cases(algorithms, parametrizations, case_configs, data_folder, kl_groupby_parametrization_cases_on_iter_folder)
  plot_kl_groupby_parametrization_cases(algorithms_, parametrizations_, case_configs, data_folder, kl_groupby_parametrization_cases_on_iter_folder)

  kl_groupby_parametrization_cases_on_time_folder = os.path.join(folder, 'kl_on_time_meanstd')
  plot_kl_groupby_parametrization_cases(algorithms, parametrizations, case_configs, data_folder, kl_groupby_parametrization_cases_on_time_folder, on_time=True)
  plot_kl_groupby_parametrization_cases(algorithms_, parametrizations_, case_configs, data_folder, kl_groupby_parametrization_cases_on_time_folder, on_time=True)
  
  kl_groupby_specific_algos_folder = os.path.join(folder, 'specific_algos')
  specific_algorithms = [algo for algo in algorithms + algorithms_ if algo[0] in ['sgd', 'adam', 'pitfalls05', 'trpKL', 'tractable', 'trpTracW2']]
  name_map = {'sgd': 'SGD', 'adam': 'ADAM', 'pitfalls05': 'Pitfalls', 'trpKL': 'TRPL-KL', 'tractable': 'Tractable', 'trpTracW2' : 'Trustable-W2'}
  plot_kl_groupby_specific_algos(specific_algorithms, case_configs, data_folder, kl_groupby_specific_algos_folder, normalize_kl=False, log_scale=True, name_map=name_map)
  # ###


  # auc_folder = os.path.join(folder, 'auc_graphs')
  # plot_auc_graphs(algorithms, parametrizations, case_configs, data_folder, auc_folder)
  # plot_auc_graphs(algorithms_, parametrizations_, case_configs, data_folder, auc_folder)

  plot_auc_tables(algorithms, parametrizations, case_configs, data_folder, auc_folder)
  plot_auc_tables(algorithms_, parametrizations_, case_configs, data_folder, auc_folder)

  
  # algos = [algorithms[i] for i in [0, 1, 2, 3, 5]]
  # algos_ = [algorithms_[i] for i in [0, 2]]
  # plot_mixed_auc_graphs(algos, parametrizations, algos_, parametrizations_, case_configs, data_folder, auc_folder)
  # #plot_mixed_auc_tables(algos, parametrizations, algos_, parametrizations_, case_configs, data_folder, auc_folder)

  # # save_latex_table(algorithms, parametrizations, case_configs, data_folder, auc_folder)
  # # save_latex_table(algorithms_, parametrizations_, case_configs, data_folder, auc_folder)

  plot_distribution(folder)