import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import numpy as np
import tensorflow as tf

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
from utils.plotting import plot_gauss_3d_animation, plot_gauss_3d, plot_gauss_multivariate_overtime, plot_gauss_gradients_2d, nd_gaussian_distribution_kl_plot, nd_gaussian_distribution_kl_plot_fixedtime
from utils.common import make_subfolders

import tr_projections.tensorflow.trust_region_layers as trust_region_layers
import tr_projections.tensorflow.vectorized_trust_region_layers as vtrl

plt.style.use('./science.mplstyle')
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['font.size'] = 32.48


### HYPERPARAMS ######
SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

iterations, batch_size, mini_batches = 400, 128, 8
alpha = 5e-2

folder = './non-contextual/figures/2d_graphs_5e-2'
os.makedirs(folder, exist_ok=True)


# which configs would be interesting to compute?
normal_configs = [
  # small correlation to large correlation
  {'initial_mean' : 0.0 * np.ones(shape=(2,)), 'initial_covariance' : np.asarray([[1.0, 0.01], [0.01, 1.0]]),
   'true_mean' : 5.0 * np.ones(shape=(2,)), 'true_covariance' : np.asarray([[1.0, 0.5], [0.5, 1.0]])},
  # huge neg correlation to large correlation
  {'initial_mean' : 0.0 * np.ones(shape=(2,)), 'initial_covariance' : np.asarray([[10.0, -5.0], [-5.0, 10.0]]),
   'true_mean' : 5.0 * np.ones(shape=(2,)), 'true_covariance' : np.asarray([[1.0, 0.5], [0.5, 1.0]])},
  # large det with small eigen [eig=[1.04721659e+02 9.54912301e-03] det=1.0000000000004174] to large correlation
  {'initial_mean' : 0.0 * np.ones(shape=(2,)), 'initial_covariance' : np.asarray([[88.56752592, -37.82297329], [-37.82297329, 16.16368182]]),
   'true_mean' : 5.0 * np.ones(shape=(2,)), 'true_covariance' : np.asarray([[1.0, 0.5], [0.5, 1.0]])},
  # large det with average eigen [eig=[0.45353926 2.20488078] det=0.9999999999999998] to large correlation
  {'initial_mean' : 0.0 * np.ones(shape=(2,)), 'initial_covariance' : np.asarray([[0.92378204, -0.77616199], [-0.77616199, 1.734638]]),
   'true_mean' : 5.0 * np.ones(shape=(2,)), 'true_covariance' : np.asarray([[1.0, 0.5], [0.5, 1.0]])},
]
######################

parametrizations = [VanillaParametrization, CholeskyParametrization, SqrtCovarianceParametrization]
algorithm_names = ['sgd', 'adam', 'natural', 'pitfalls05', 'pitfalls10', 'trpFrob', 'trpW2', 'trpKL', 'gaussnewton', 'trpW2Force']
algorithms = [
  (sgd_gaussian_optimization, {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (adam_gaussian_optimization, {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (natural_gaussian_optimization, {'delta' : 0.001, 'iterations' : 50, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (pitfalls_gaussian_optimization, {'alpha' : alpha, 'beta' : 0.5, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (pitfalls_gaussian_optimization, {'alpha' : alpha, 'beta' : 1.0, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (trp_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.FrobProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (trp_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (trp_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.KLProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (gaussnewton_gaussian_optimization, {'alpha' : alpha, 'iterations' : 50, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (trp_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ForceProjectionLayer(mean_bound=0.1, covariance_bound=0.1), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
]

# parametrizations_ = [TractableCholeskyParametrization, ApproximateTractableCholeskyParametrization]
parametrizations_ = [ApproximateTractableCholeskyParametrization]
algorithm_names_ = ['tractable', 'trpTracFrob', 'trpTracW2', 'trpTracKL', 'trpTracCutW2']
algorithms_ = [
  (tractable_gaussian_optimization, {'alpha' : alpha, 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (traptable_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.FrobProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (traptable_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.W2ProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
  (traptable_gaussian_optimization, {'alpha' : alpha, 'proj_layer' : trust_region_layers.KLProjectionLayer(mean_bound=0.1, covariance_bound=1.0), 'iterations' : iterations, 'batch_size' : batch_size, 'mini_batches' : mini_batches}),
]



### RUN ALGORITHMS IF NEEDED
def run_algorithms(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      # log figure
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        if not os.path.exists(save_name):
          with tf.device('/cpu:0'):
            log = algo_fnc(**normal_config, phi=parametrization(), **params)
          np.save(save_name, log)

make_subfolders(folder, ['data'])

run_algorithms(algorithms, algorithm_names, normal_configs, parametrizations)
run_algorithms(algorithms_, algorithm_names_, normal_configs, parametrizations_)


### SINGLE PLOTS
def plot_singles(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      # log figure
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        means = np.asarray([log_['mean'] for log_ in log])
        covariances = np.asarray([log_['covariance'] for log_ in log])
        iterations = covariances.shape[0]

        # single graph plots
        fig_singlegraph, ax_singlegraph = plt.subplots(1, 1)
        param_patch = mlines.Line2D([], [], markerfacecolor='none', markeredgecolor='k', marker='o', linestyle='None', label=parametrization.__name__.removesuffix('Parametrization'))
        plot_gauss_gradients_2d(means, covariances,
                                initial_mean, initial_covariance,
                                true_mean, true_covariance,
                                equal_aspect=False, ax=ax_singlegraph)
        ax_singlegraph.legend(handles=[param_patch], loc='upper left')
        
        initial_covariance_str = f'{initial_covariance}'.replace('\n', '')
        true_covariance_str = f'{true_covariance}'.replace('\n', '')
        # fig_singlegraph.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance_str}\nμ_true={true_mean}, σ²_true={true_covariance_str} | iter={iterations}', fontsize='small')

        fig_singlegraph.set_tight_layout(True)
        fig_singlegraph.savefig(f'{folder}/singlegraph/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.svg', dpi=1200)

      plt.close('all')

make_subfolders(folder, ['singlegraph'])




### SINGLE COVARIANCE MATRIX OVER TIME
def plot_singles_overtime(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      # log figure
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        means = np.asarray([log_['mean'] for log_ in log])
        covariances = np.asarray([log_['covariance'] for log_ in log])

        # single graph plots
        fig_singlegraph, axes_singlegraph = plt.subplots(covariances.shape[1] + 2, covariances.shape[2])
        plot_gauss_multivariate_overtime(means, covariances, true_mean, true_covariance, axes_singlegraph)

        initial_covariance_str = f'{initial_covariance}'.replace('\n', '')
        true_covariance_str = f'{true_covariance}'.replace('\n', '')
        # fig_singlegraph.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance_str}\nμ_true={true_mean}, σ²_true={true_covariance_str} | {parametrization.__name__.removesuffix("Parametrization")}', fontsize='small')

        fig_singlegraph.set_tight_layout(True)
        fig_singlegraph.savefig(f'{folder}/singlegraph_overtime/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.svg', dpi=1200)

      plt.close('all')

make_subfolders(folder, ['singlegraph_overtime'])




### SINGLE PLOTS WITH 3D ANIMATION
def plot_singles_3d_animation(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      # log figure
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        means = np.asarray([log_['mean'] for log_ in log])
        covariances = np.asarray([log_['covariance'] for log_ in log])

        # single graph plots
        # single graph plots
        ani = plot_gauss_3d_animation(means, covariances, None)
        ani.save(f'{folder}/singlegraph_3d_animation/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.gif', writer='imagemagick', fps=10)
        
        #fig_singlegraph.set_tight_layout(True)
        #plt.show()



        #quit() # TODO REMOVE
        #fig_singlegraph.set_tight_layout(True)
        #fig_singlegraph.savefig(f'{folder}/singlegraph/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.svg', dpi=1200)

      plt.close('all')

make_subfolders(folder, ['singlegraph_3d_animation'])

#plot_singles_3d_animation(algorithms, algorithm_names, normal_configs, parametrizations)
#plot_singles_3d_animation(algorithms_, algorithm_names_, normal_configs, parametrizations_)


"""
### SHARED PLOTS PER ALGO
def plot_shared_per_algo(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()

      fig_graph, ax_graph = plt.subplots(1, 1)
      
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        means = np.asarray([log_['mean'].squeeze() for log_ in log])
        covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])

        # multi graph plots
        plot_gauss_gradients_1d_line(means, covariances,
                                    initial_mean, initial_covariance, 
                                    true_mean, true_covariance, 
                                    label=parametrization.__name__.removesuffix('Parametrization'), 
                                    restrict_area=True, log_scale=False, ax=ax_graph)

      fig_graph.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance}\nμ_true={true_mean}, σ²_true={true_covariance} | iter={iterations}')
      fig_graph.set_tight_layout(True)

      ax_graph.legend()
      ax_graph.ticklabel_format(axis='x', style='plain', useOffset=False)
      
      fig_graph.savefig(f'{folder}/{algorithm_names[algo_id]}_{normal_id}.svg', dpi=1200)
      plt.close('all')

plot_shared_per_algo(algorithms, algorithm_names, normal_configs, parametrizations)
plot_shared_per_algo(algorithms_, algorithm_names_, normal_configs, parametrizations_)


### LOG SHARED PLOTS PER ALGO
def plot_log_shared_per_algo(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      fig_loggraph, ax_loggraph = plt.subplots(1, 1)      

      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        means = np.asarray([log_['mean'].squeeze() for log_ in log])
        covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])

        plot_gauss_gradients_1d_line(means, covariances, 
                                    initial_mean, initial_covariance, 
                                    true_mean, true_covariance, 
                                    label=parametrization.__name__.removesuffix('Parametrization'), 
                                    restrict_area=False, log_scale=True, ax=ax_loggraph)

      fig_loggraph.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance}\nμ_true={true_mean}, σ²_true={true_covariance} | iter={iterations}')
      fig_loggraph.set_tight_layout(True)

      ax_loggraph.legend()
      ax_loggraph.ticklabel_format(axis='x', style='plain', useOffset=False)
      
      fig_loggraph.savefig(f'{folder}/loggraph/{algorithm_names[algo_id]}_{normal_id}_log.svg', dpi=1200)
      plt.close('all')

make_subfolders(folder, ['loggraph'])

plot_log_shared_per_algo(algorithms, algorithm_names, normal_configs, parametrizations)
plot_log_shared_per_algo(algorithms_, algorithm_names_, normal_configs, parametrizations_)

"""

### LOSS SHARED PLOTS PER ALGO
def plot_loss_per_algo(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      fig_loss, ax_loss = plt.subplots(1, 1)

      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9
        means = np.asarray([log_['mean'].squeeze() for log_ in log])
        covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])
        iterations = covariances.shape[0]
        
        nd_gaussian_distribution_kl_plot(means, covariances,
                                        true_mean, true_covariance,
                                        label=parametrization.__name__.removesuffix('Parametrization'),
                                        ax=ax_loss)
      ax_loss2 = ax_loss.twiny()
      ax_loss2.set_xticks([x for x in ax_loss.get_xticks() if x < times.size and x >= 0])
      ax_loss2.set_xbound((0, times.size))
      ax_loss2.set_xticklabels([round(times[int(x)], 2) for x in ax_loss2.get_xticks()])
      ax_loss2.set_xlabel('t [s]')

      
      initial_covariance_str = f'{initial_covariance}'.replace('\n', '')
      true_covariance_str = f'{true_covariance}'.replace('\n', '')
      # fig_loss.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance_str}\nμ_true={true_mean}, σ²_true={true_covariance_str} | iter={iterations}', fontsize='small')
      fig_loss.set_tight_layout(True)

      ax_loss.legend()
      ax_loss.ticklabel_format(axis='x', style='plain', useOffset=False)
      
      fig_loss.savefig(f'{folder}/lossgraph/{algorithm_names[algo_id]}_{normal_id}_loss.svg', dpi=1200)
      plt.close('all')

make_subfolders(folder, ['lossgraph'])




### FIXED TIME LOSS PLOTS PER ALGO
def plot_fixedtime_loss_per_algo(algorithms, algorithm_names, normal_configs, parametrizations):
  for algo_id, (algo_fnc, params) in enumerate(algorithms):
    for normal_id, normal_config in enumerate(normal_configs):
      initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()
      
      fig_loss, ax_loss = plt.subplots(1, 1)
      
      # log figure
      for parametrization in parametrizations:
        save_name = f'{folder}/data/{algorithm_names[algo_id]}_{parametrization.__name__}_{normal_id}.npy'
        log = np.load(save_name, allow_pickle=True)

        times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9
        means = np.asarray([log_['mean'].squeeze() for log_ in log])
        covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])
        iterations = covariances.shape[0]
        
        nd_gaussian_distribution_kl_plot_fixedtime(times, means, covariances,
                                        true_mean, true_covariance,
                                        label=parametrization.__name__.removesuffix('Parametrization'),
                                        ax=ax_loss)

      initial_covariance_str = f'{initial_covariance}'.replace('\n', '')
      true_covariance_str = f'{true_covariance}'.replace('\n', '')
      # fig_loss.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance_str}\nμ_true={true_mean}, σ²_true={true_covariance_str} | iter={iterations}', fontsize='small')
      fig_loss.set_tight_layout(True)

      ax_loss.legend()
      ax_loss.ticklabel_format(axis='x', style='plain', useOffset=False)
      
      fig_loss.savefig(f'{folder}/timelossgraph/{algorithm_names[algo_id]}_{normal_id}_loss.svg', dpi=1200)
      plt.close('all')

make_subfolders(folder, ['timelossgraph'])




### FIXED TIME LOSS PLOTS PER PARAM
def plot_fixedtime_loss_per_param(algorithms, algorithm_names, normal_configs, parametrizations):
  for normal_id, normal_config in enumerate(normal_configs):
    initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()

    for parametrization in parametrizations:
      
      fig_loss, ax_loss = plt.subplots(1, 1)

      for algo_name in algorithm_names:
        save_name = f'{folder}/data/{algo_name}_{parametrization.__name__}_{normal_id}.npy'
        if not os.path.exists(save_name):
          continue
        log = np.load(save_name, allow_pickle=True)

        times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9
        means = np.asarray([log_['mean'].squeeze() for log_ in log])
        covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])
        iterations = covariances.shape[0]
        
        nd_gaussian_distribution_kl_plot_fixedtime(times, means, covariances,
                                        true_mean, true_covariance,
                                        label=algo_name,
                                        ax=ax_loss)
      
      initial_covariance_str = f'{initial_covariance}'.replace('\n', '')
      true_covariance_str = f'{true_covariance}'.replace('\n', '')
      # fig_loss.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance_str}\nμ_true={true_mean}, σ²_true={true_covariance_str} | iter={iterations}', fontsize='small')
      fig_loss.set_tight_layout(True)

      ax_loss.legend()
      ax_loss.ticklabel_format(axis='x', style='plain', useOffset=False)
      
      fig_loss.savefig(f'{folder}/paramtimelossgraph/{parametrization.__name__.removesuffix("Parametrization")}_{normal_id}_loss.svg', dpi=1200)
      plt.close('all')


make_subfolders(folder, ['paramtimelossgraph'])






plot_singles(algorithms, algorithm_names, normal_configs, parametrizations)
plot_singles(algorithms_, algorithm_names_, normal_configs, parametrizations_)
# plot_singles_overtime(algorithms, algorithm_names, normal_configs, parametrizations)
# plot_singles_overtime(algorithms_, algorithm_names_, normal_configs, parametrizations_)
plot_loss_per_algo(algorithms, algorithm_names, normal_configs, parametrizations)
plot_loss_per_algo(algorithms_, algorithm_names_, normal_configs, parametrizations_)
# plot_fixedtime_loss_per_algo(algorithms, algorithm_names, normal_configs, parametrizations)
# plot_fixedtime_loss_per_algo(algorithms_, algorithm_names_, normal_configs, parametrizations_)
# plot_fixedtime_loss_per_param(algorithms + algorithms_, algorithm_names + algorithm_names_, normal_configs, parametrizations + parametrizations_)

quit()

### REPRESENTABLE FIXED TIME LOSS PLOTS PER PARAM (ONE PER ALGO)
"""
make_subfolders(folder, ['bestof_plots'])

best_combinations = [
  [
    ('sgd', SqrtCovarianceParametrization),
    ('natural', LogDiagCovarianceParametrization),
    ('pitfalls10', SqrtCovarianceParametrization),
    ('trpW2', LogDiagCovarianceParametrization),
    ('gaussnewton', LogDiagCovarianceParametrization),
    ('tractable', SqrtCovarianceParametrization),
  ],
  [
    ('sgd', LogSqrtDiagCovarianceParametrization),
    ('natural', SoftplusDiagCovarianceParametrization),
    ('pitfalls10', LogSqrtDiagCovarianceParametrization),
    ('trpW2', SqrtCovarianceParametrization),
    ('gaussnewton', LogDiagCovarianceParametrization),
    ('tractable', SqrtCovarianceParametrization),
  ],
  [
    ('sgd', SoftplusDiagCovarianceParametrization),
    ('natural', DiagCovarianceParametrization),
    ('pitfalls10', LogSqrtDiagCovarianceParametrization),
    ('trpW2', SqrtCovarianceParametrization),
    ('gaussnewton', LogSqrtDiagCovarianceParametrization),
    ('tractable', SqrtCovarianceParametrization),
  ],
]

for normal_id, normal_config in enumerate(normal_configs):
  initial_mean, initial_covariance, true_mean, true_covariance = normal_config.values()

  fig_loss, ax_loss = plt.subplots(1, 1)

  for algo_name, parametrization in best_combinations[normal_id]:
    save_name = f'{folder}/data/{algo_name}_{parametrization.__name__}_{normal_id}.npy'
    log = np.load(save_name, allow_pickle=True)
    
    times = (np.asarray([log_['t'] for log_ in log]) - log[0]['t']) / 1e9
    means = np.asarray([log_['mean'].squeeze() for log_ in log])
    covariances = np.asarray([log_['covariance'].squeeze() for log_ in log])
    
    nd_gaussian_distribution_kl_plot_fixedtime(times, means.reshape((-1, 1)), covariances.reshape((-1, 1, 1)),
                                        true_mean, true_covariance,
                                        label=f'{algo_name} {parametrization.__name__.removesuffix("Parametrization")}',
                                        ax=ax_loss)
      
  fig_loss.suptitle(f'μ_init={initial_mean}, σ²_init={initial_covariance}\nμ_true={true_mean}, σ²_true={true_covariance}')
  fig_loss.set_tight_layout(True)

  ax_loss.legend()
  ax_loss.ticklabel_format(axis='x', style='plain', useOffset=False)
      
  fig_loss.savefig(f'{folder}/bestof_plots/{normal_id}_loss.svg', dpi=1200)
  plt.close('all')

"""