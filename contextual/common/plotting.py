import numpy as np
import scipy.stats as st

def _confidence_to_z(confidence):
  return st.norm.ppf((1 + confidence) / 2)

# NOTE: either plot (mean, CI) or (median, quartiles?)

# curves = [N x I x 2] with N curves and I iterations , 2 x I must be numpy array
def plot_median_confi_curve(curves, ax, plot_singles=True, mean_label='', color='red', smoothing=2, confidence=0.85):
  max_steps = np.asarray([curve[-1, 0] for curve in curves]).max()

  xs = np.arange(start=0, stop=max_steps, step=1)
  ys_ = np.asarray([np.interp(xs, curve[:, 0], curve[:, 1]) for curve in curves])

  def smooth(x, y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return x[:-box_pts + 1], y_smooth

  if plot_singles:
    # plot normal curves in grayish
    for ys in ys_:
      ax.plot(xs, ys, color='#0f0f0f', alpha=0.05)

  # plot mean and std
  xs_, mean_curve = smooth(xs, np.median(ys_, axis=0), smoothing)
  ax.plot(xs_, mean_curve, color=color, label=mean_label)

  n = len(curves)
  z = _confidence_to_z(confidence)
  q = 0.5 # quantile

  j = int(n * q - z * np.sqrt(n * q * (1 - q)))
  k = int(n * q + z * np.sqrt(n * q * (1 - q)))

  sorted_ys_ = np.sort(ys_, axis=0)
  lower_ci, upper_ci = sorted_ys_[j, :], sorted_ys_[k, :]

  xs_, lower_ci_curve = smooth(xs, lower_ci, smoothing)
  xs_, upper_ci_curve = smooth(xs, upper_ci, smoothing)
  ax.fill_between(xs_, lower_ci_curve, upper_ci_curve, alpha=0.15, color=color)


# curves = [N x I x 2] with N curves and I iterations , 2 x I must be numpy array
def plot_mean_confi_curve(curves, ax, plot_singles=True, mean_label='', color='red', smoothing=2, confidence=0.95):
  max_steps = np.asarray([curve[-1, 0] for curve in curves]).max()

  xs = np.arange(start=0, stop=max_steps, step=1)
  ys_ = np.asarray([np.interp(xs, curve[:, 0], curve[:, 1]) for curve in curves])

  def smooth(x, y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return x[:-box_pts + 1], y_smooth

  if plot_singles:
    # plot normal curves in grayish
    for ys in ys_:
      ax.plot(xs, ys, color='#0f0f0f', alpha=0.05)

  # plot mean and confidence interval
  num_samples = len(curves)
  mean, std = np.mean(ys_, axis=0), np.std(ys_, axis=0)
  ci = _confidence_to_z(confidence) * std / np.sqrt(num_samples)
  
  xs_, mean_curve = smooth(xs, mean, smoothing)
  ax.plot(xs_, mean_curve, color=color, label=mean_label)

  xs_, ci_curve = smooth(xs, ci, smoothing)
  ax.fill_between(xs_, mean_curve - ci_curve, mean_curve + ci_curve, alpha=0.15, color=color)

def plot_confidence_ellipsoid(mean, cov, ax, confidence=0.95, **kwargs):
  u = np.linspace(0, 2 * np.pi, 33)
  v = np.linspace(0, np.pi, 33)

  r = np.sqrt(st.chi2.ppf(confidence, df=3))

  x = r * np.outer(np.cos(u), np.sin(v))
  y = r * np.outer(np.sin(u), np.sin(v))
  z = r * np.outer(np.ones_like(u), np.cos(v))

  L = np.linalg.cholesky(cov)
  ellipsoid = (L @ np.stack((x, y, z), 0).reshape(3, -1) + mean).reshape(3, *x.shape)

  return ax.plot_wireframe(*ellipsoid, rstride=4, cstride=4, **kwargs)