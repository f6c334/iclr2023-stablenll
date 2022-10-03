
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import seaborn as sns

from utils.common import kl_mvn, w2_mvn, frob_mvn, rmse_mvn, multivariate_normal_pdf


def plot_gauss_gradients_1d_dots(means, covariances, initial_mean, initial_covariance, true_mean, true_covariance, ax):
  ax.axvline(x=true_mean, color='r', alpha=0.35)
  ax.axhline(y=true_covariance, color='r', alpha=0.35)

  colors = cm.copper(np.linspace(0, 1, len(means)))
  ax.scatter(means, covariances, color=colors)

  if any(np.isnan(covariances)):
    means_, covariances_ = np.argwhere(np.isnan(means)), np.argwhere(np.isnan(covariances))
    min_idx = np.asarray([means_, covariances_]).min()
        
    # annotate smth at min_idx - 1
    ax.plot(means[min_idx - 1], covariances[min_idx - 1], 'or')
  
  ax.set_xlabel('mean')
  ax.set_ylabel('variance')


def plot_gauss_gradients_1d_line(means, covariances, initial_mean, initial_covariance, true_mean, true_covariance, label, restrict_area, log_scale, ax):
  ax.axvline(x=true_mean, color='r', alpha=0.05)
  ax.axhline(y=true_covariance, color='r', alpha=0.05)
  
  ax.plot(initial_mean, initial_covariance, 'ok')
  ax.plot(means, covariances, '--', label=label, alpha=0.75)

  if any(np.isnan(covariances)):
    means_, covariances_ = np.argwhere(np.isnan(means)), np.argwhere(np.isnan(covariances))
    min_idx = np.asarray([means_, covariances_]).min()
        
    # annotate smth at min_idx - 1
    ax.plot(means[min_idx - 1], covariances[min_idx - 1], 'xr')
  else:
    ax.plot(means[-1], covariances[-1], 'xk')

  
  
  if restrict_area and covariances.max() > 20:
    ax.set_xlim(left=-1, right=6)
    ax.set_ylim(bottom=-0.5, top=20)
  
  ax.set_xlabel('mean')
  ax.set_ylabel('variance')

  if log_scale:
    ax.set_yscale('log')
    ax.set_ylabel('log variance')





# input data of shape [B x D]
def parallel_coordinates_plot(host, data, colors, y_scale_equal = False):
  num_lines, dims = data.shape

  ynames = [str(i) for i in range(dims)]
  if type(colors) is str:
    colors = [colors for i in range(data.shape[0])]

  # get scaling factors
  ymins, ymaxs = data.min(axis=0), data.max(axis=0)
  dys = ymaxs - ymins
  
  ymins -= dys * 0.05  # add 5% padding below and above
  ymaxs += dys * 0.05

  eqs = ymins == ymaxs
  ymins[eqs] -= 0.5
  ymaxs[eqs] += 1.0
  dys = ymaxs - ymins

  if y_scale_equal:
    ymins = np.full_like(ymins, ymins.min())
    ymaxs = np.full_like(ymaxs, ymaxs.max())
    dys = ymaxs - ymins

  # transform all data to be compatible with the main axis
  zs = np.zeros_like(data)
  zs[:, 0] = data[:, 0]
  zs[:, 1:] = (data[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


  axes = [host] + [host.twinx() for i in range(data.shape[1] - 1)]
  for i, ax in enumerate(axes):
    #if not np.any(np.isnan(data)):
    if y_scale_equal:
      ax.set_ylim(ymins.min(), ymaxs.max())
    else:  
      ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
      ax.spines['left'].set_visible(False)
      ax.yaxis.set_ticks_position('right')
      ax.spines["right"].set_position(("axes", i / (data.shape[1] - 1)))

  host.set_xlim(0, data.shape[1] - 1)
  host.set_xticks(range(data.shape[1]))
  host.set_xticklabels(ynames, fontsize=14)
  host.tick_params(axis='x', which='major', pad=7)
  host.spines['right'].set_visible(False)
  host.xaxis.tick_top()

  for j in range(num_lines):
    host.plot(range(dims), zs[j,:], c=colors[j])


def plot_nd_gaussian_kl(means, covariances, true_mean, true_covariance, label, ax, times=None, normalize_kl=False, log_scale=False):
  kl = np.asarray([kl_mvn(mean, covariance, true_mean, true_covariance) 
                  for mean, covariance in zip(means, covariances)])
  metric = 1 - np.exp(- kl) if normalize_kl else kl
  
  label_prefix = 'NKL' if normalize_kl else 'KL'

  if times is None:
    ax.plot(range(len(metric)), metric, label=f'{label_prefix} {label}')
    ax.set_xlabel('Iterations')
  else:
    ax.plot(times, metric, label=f'{label_prefix} {label}')
    ax.set_xlabel('t [s]')
  
  if log_scale:
    ax.set_yscale('symlog')
    
  ax.set_ylabel(label_prefix)

  
  

def nd_gaussian_distribution_kl_plot(means, covariances, true_mean, true_covariance, label, ax, normalize_kl=False):
  kl = [kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]
  
  if normalize_kl:
    nkl = 1 - np.exp(- np.asarray(kl))
    ax.plot(range(len(nkl)), kl, label='NKL ' + label)
  else:
    ax.plot(range(len(kl)), kl, label='KL ' + label)
    ax.set_yscale('log')
  
  ax.set_xlabel('Iterations')

def nd_gaussian_distribution_kl_plot_fixedtime(times, means, covariances, true_mean, true_covariance, label, ax, normalize_kl=False):
  kl = [kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]

  if normalize_kl:
    nkl = 1 - np.exp(- np.asarray(kl))
    ax.plot(times, nkl, label=label)
  else:
    ax.plot(times, kl, label=label)
    ax.set_yscale('log')
  ax.set_xlabel('t [s]')
  ax.set_ylabel('KL' if not normalize_kl else 'NKL')

def nd_gaussian_distribution_metrics_plot(means, covariances, true_mean, true_covariance, ax):
  frob = [frob_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]
  w2 = [w2_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]
  kl = [kl_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]
  rmse = [rmse_mvn(mean, covariance, true_mean, true_covariance) for mean, covariance in zip(means, covariances)]

  ax.plot(range(len(frob)), frob, label='Frob')
  ax.plot(range(len(w2)), w2, label='W2')
  ax.plot(range(len(kl)), kl, label='KL')

  ax2 = ax.twinx()
  ax2.plot(range(len(rmse)), [m_rmse for m_rmse, _ in rmse], ':k', label='RMSE-Mean')
  ax2.plot(range(len(rmse)), [c_rmse for _, c_rmse in rmse], '--k', label='RMSE-Covariance')

  ax.set_title('Distribution Metrics: True vs Estimated')
  ax.set_yscale('log')
  ax.legend(loc='upper right')
  ax.set_ylabel('Solid Line Scale')
  ax.set_xlabel('Iterations')
  ax2.set_ylabel('Dotted Line Scale')
  ax2.legend(loc='upper center')
  ax2.set_yscale('log')

def nd_gaussian_covariance_heatmap_plot(covariance, vmin, vmax, ax, cbar_ax):
  mask = np.zeros_like(covariance)
  mask[np.triu_indices_from(mask)] = True

  sns.heatmap(covariance, square=True, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cbar_ax)

def nd_gaussian_visualization_plot(means, covariances, true_mean, true_covariance, num_means_parallel_coords, num_covariances_heatmaps, title=''):
  batch, dim = means.shape
  
  # Set up the axes with gridspec
  fig = plt.figure(figsize=(16, 9))
  fig.suptitle(title, fontsize=16)
  grid = plt.GridSpec(6, num_covariances_heatmaps + 1, hspace=0.2, wspace=0.2)


  ### mean plot
  # parallel coordinates over all dimensions and the eqally spaced samples during training
  mean_ax = fig.add_subplot(grid[0:4, :int(0.5 * grid.ncols)])

  means_idx = np.linspace(0, batch - 1, num=num_means_parallel_coords, dtype=np.int)
  pc_means = means[means_idx]
  pc_means = np.concatenate([pc_means, true_mean[np.newaxis]])
  colors = list(cm.copper(np.linspace(0, 1, num_means_parallel_coords))) + ['red']

  parallel_coordinates_plot(mean_ax, pc_means, colors, y_scale_equal=True)

  mean_ax.set_title('Means during Training')


  ### covariance plot
  # covariance rmse, kldiv, ...
  covmetrics_ax = fig.add_subplot(grid[0:4, 1 + int(0.5 * grid.ncols):])

  nd_gaussian_distribution_metrics_plot(means, covariances, true_mean, true_covariance, covmetrics_ax)

  # covariance heatmaps plots
  hm_covariances = covariances[::int(batch / num_covariances_heatmaps) + 1]

  cbar_ax = fig.add_axes([0.915, 0.085, 0.015, 0.25])

  sns.set(font_scale=0.5)

  true_cov_ax = fig.add_subplot(grid[4, 0])
  nd_gaussian_covariance_heatmap_plot(true_covariance, vmin=covariances.min(), vmax=covariances.max(), ax=true_cov_ax, cbar_ax=cbar_ax)
  true_cov_ax.set_title('True Covariance')

  init_cov_ax = fig.add_subplot(grid[5, 0])
  nd_gaussian_covariance_heatmap_plot(covariances[0], vmin=covariances.min(), vmax=covariances.max(), ax=init_cov_ax, cbar_ax=cbar_ax)
  init_cov_ax.set_title('Initial Covariance')

  for i in range(0, num_covariances_heatmaps) :
    hm_ax = fig.add_subplot(grid[4:, i + 1])
    
    covariance = hm_covariances[i]
    #covariance = np.abs(covariance) # TODO: nice scaling for this...

    nd_gaussian_covariance_heatmap_plot(covariance, vmin=covariances.min(), vmax=covariances.max(), ax=hm_ax, cbar_ax=cbar_ax)
  
  sns.reset_orig()


def plot_gauss_gradients_2d(est_means, est_covariances, initial_mean, initial_covariance, true_mean, true_covariance, equal_aspect, ax):
  colors = cm.copper(np.linspace(0, 1, len(est_means)))
  ax.scatter(est_means[:, 0], est_means[:, 1], s=10.0, color=colors, alpha=0.75)

  for i in range(len(est_means)):
    plot_diag_confidence_ellipse(est_means[i], est_covariances[i], ax, edgecolor=colors[i], linewidth=1.0, alpha=0.1)
  
  plot_diag_confidence_ellipse(est_means[0], est_covariances[0], ax, edgecolor='gray', linewidth=1.0, linestyle=':', alpha=0.5)

  ax.scatter(est_means[-1, 0], est_means[-1, 1], s=12.0, linestyle=':', edgecolor='g', alpha=0.5)
  plot_diag_confidence_ellipse(est_means[-1], est_covariances[-1], ax, edgecolor='g', linewidth=1.0, linestyle=':', alpha=0.5)

  ax.scatter(true_mean[0], true_mean[1], facecolors='none', edgecolors='r', alpha=0.5)
  plot_diag_confidence_ellipse(true_mean, true_covariance, ax, edgecolor='r', linewidth=1.0, linestyle='--', alpha=0.5)
  
  if equal_aspect:
    ax.set_aspect('equal', adjustable='box')

  ax.set_xlabel('x_1')
  ax.set_ylabel('x_2')

def plot_diag_confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
  pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

  ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)

  ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,facecolor=facecolor, **kwargs)

  mean_x, mean_y = mean
  scale_x = np.sqrt(cov[0, 0]) * n_std
  scale_y = np.sqrt(cov[1, 1]) * n_std


  transf = transforms.Affine2D() \
    .rotate_deg(45) \
    .scale(scale_x, scale_y) \
    .translate(mean_x, mean_y)

  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)


def plot_gauss_multivariate_overtime(est_means, est_covariances, true_mean, true_covariance, axes):
  mean_axes, cov_axes, eig_axes = axes[0, :], axes[1:-1, :], axes[-1, :]

  for i in range(est_means.shape[-1]):
    mean_axes[i].plot(range(len(est_means[:, i])), est_means[:, i])

    mean_axes[i].axhline(y=true_mean[i], color='r', alpha=0.25)

    mean_axes[i].set_xlabel('Iterations')
    mean_axes[i].set_ylabel('$\mu_{' + str(i) + '}$')

  for i in range(est_means.shape[-1]):
    for j in range(est_means.shape[-1]):
      cov_axes[i][j].plot(range(len(est_covariances[:, i, j])), est_covariances[:, i, j], linestyle='--')
      
      cov_axes[i][j].axhline(y=true_covariance[i, j], color='r', alpha=0.25)

      cov_axes[i][j].set_yscale('symlog')
      cov_axes[i][j].set_xlabel('Iterations')
      cov_axes[i][j].set_ylabel('$\Sigma_{' + str(i) + str(j) + '}$')
  
  eigs = np.linalg.eigvals(np.nan_to_num(est_covariances))
  for i in range(est_means.shape[-1]):
    eig_axes[i].plot(range(len(est_covariances[:, i, j])), eigs[:, i])
    
    eig_axes[i].set_xlabel('Iterations')
    eig_axes[i].set_ylabel('$\lambda_{' + str(i) + '}$')
    


def plot_gauss_3d(mean, covariance, ax, meshsize=30, x_range=(-10, 10), y_range=(-10, 10)):
  x = np.linspace(*x_range, meshsize)
  y = np.linspace(*y_range, meshsize)
  x, y = np.meshgrid(x, y)
  z = np.zeros((meshsize, meshsize))
  
  for i in range(meshsize):
    for j in range(meshsize):
      z[i, j] = multivariate_normal_pdf(np.asarray([x[i, j], y[i, j]]), mean, covariance)

  return ax.plot_surface(x, y, z, cmap='jet')


def plot_gauss_3d_animation(means, covariances, ax):
  N = 30 # Meshsize
  fps = 10 # frame per sec
  frn, dim = means.shape # frame number of the animation

  # determine x and y range
  mu_mean = np.mean(means, axis=0)
  
  # B x D
  mean_min = np.min(means, axis=0)
  mean_max = np.max(means, axis=0)

  x_range = (mean_min[0] - 5, mean_max[0] + 5)
  y_range = (mean_min[1] - 5, mean_max[1] + 5)
  
  def update_plot(frame_number, plot, dummy):
    plot[0].remove()
    plot[0] = plot_gauss_3d(means[frame_number], covariances[frame_number], ax, N, x_range, y_range)


  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  plot = [plot_gauss_3d(means[0], covariances[0], ax)]
  ax.set_zlim(0,1.1)
  ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(plot, None), interval=1000/fps)
  return ani

def plot_3d_func(func, x_range, y_range, mesh_size, ax, cmap='magma'):
  x, y = np.linspace(*x_range, mesh_size), np.linspace(*y_range, mesh_size)
  x, y = np.meshgrid(x, y)
  z = np.zeros((mesh_size, mesh_size))
  
  for i in range(mesh_size):
    for j in range(mesh_size):
      z[i, j] = func(x[i, j], y[i, j])

  return ax.plot_surface(x, y, z, cmap=cmap)

def plot_arrow(origin, vector, ax, **kwargs):
  ax.plot(*np.stack([origin, vector]).T, **kwargs)



# curves = [N x 2 x I] with N curves and I iterations , 2 x I must be numpy array
def plot_mean_std_curve(curves, ax, plot_singles=True, mean_label='', color='red', smoothing=2):
  max_steps = np.asarray([curve[-1, 0] for curve in curves]).max()

  xs = np.arange(start=0, stop=max_steps, step=1)
  ys_ = np.asarray([np.interp(xs, curve[:, 0], curve[:, 1]) for curve in curves])

  def smooth(x, y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return x[:- box_pts + 1], y_smooth

  if plot_singles:
    # plot normal curves in grayish
    for ys in ys_:
      ax.plot(xs, ys, color='#0f0f0f', alpha=0.05)

  # plot mean and std
  xs_, mean_curve = smooth(xs, np.mean(ys_, axis=0), smoothing)
  ax.plot(xs_, mean_curve, color=color, label=mean_label)

  xs_, std_curve = smooth(xs, np.std(ys_, axis=0), smoothing)
  ax.fill_between(xs_, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=color)