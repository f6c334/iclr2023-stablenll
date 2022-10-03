from typing import List, Tuple

import numpy as np

from common import utils


def sample_wrapper(true_function, x):
  mean, covariance = true_function(x)
  sample = utils.sample_batch_mvn(mean, covariance)

  return sample, mean, covariance


###############################################################################
### 1D ########################################################################
def sf_constant(x: np.ndarray, value: float = 0.0, noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
  """A simple 1D constant output function with constant noise.

  Args:
      x (np.ndarray): [B] or [B x 1]
      value (float, optional): []. Defaults to 0.0.
      noise (float, optional): Standard deviation []. Defaults to 0.01.

  Returns:
      Tuple[np.ndarray, np.ndarray]: Means and covarianes at x. [B x 1], [B x 1 x 1]
  """

  def f(x):
    return np.full_like(x, fill_value=value)

  variance = np.full_like(x, fill_value=np.square(noise))

  return f(x).reshape(-1, 1), variance.reshape(-1, 1, 1)


def sf_n_constant(x: np.ndarray,
                  x_range: Tuple[float, float] = (-1.0, 1.0),
                  y_range: Tuple[float, float] = (-1.0, 1.0),
                  noises: List[float] = [0.1, 1.0]) -> Tuple[np.ndarray, np.ndarray]:
  """Piecewise 1D constant function within range of x and y as defined by parameters. Noise is equidistantly adapted
     to whatever is given in the noises parameter.

  Args:
      x (np.ndarray): [B] or [B x 1]
      x_range (Tuple[float, float], optional): Range of x, between which mean and variance is varied.
        Defaults to (-1.0, 1.0).
      y_range (Tuple[float, float], optional): Range of y, between which mean is varied.
        Defaults to (-1.0, 1.0).
      noises (List[float], optional): List of standard deviations that are applied in x range, equidistant. 
        Defaults to [0.1, 1.0].

  Returns:
      Tuple[np.ndarray, np.ndarray]: Means and covarianes at x. [B x 1], [B x 1 x 1]
  """
  space = (x_range[1] - x_range[0]) / len(noises)
  condlist = [(x <= x_ + space) & (x >= x_) for x_ in np.linspace(*x_range, len(noises), endpoint=False)]

  def f(x):
    return np.piecewise(x, condlist=condlist, funclist=np.linspace(*y_range, len(noises), endpoint=False))

  variance = np.piecewise(x, condlist=condlist, funclist=np.square(noises))

  return f(x).reshape(-1, 1), variance.reshape(-1, 1, 1)


def sf_pitfalls_sinusoidal(x: np.ndarray, offset: float = 0.0, noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
  """Implementing the 1D sinus function with constant noise presented in
    
    Seitzer, Maximilian, et al. 
    "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." 
    arXiv preprint arXiv:2203.09168 (2022).

  Args:
      x (np.ndarray): [B] or [B x 1]
      offset (float, optional): Y-offset []. Defaults to 0.0.
      noise (float, optional): Standard deviation. Defaults to 0.01.

  Returns:
      Tuple[np.ndarray, np.ndarray]: Means and covarianes at x. [B x 1], [B x 1 x 1]
  """

  def f(x):
    return 0.4 * np.sin(2 * np.pi * x) + offset

  variance = np.full_like(x, fill_value=np.square(noise))

  return f(x).reshape(-1, 1), variance.reshape(-1, 1, 1)


def sf_detlefsen_sinusoidal(x: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
  """Implementing the 1D sinusoidal function with variable noise presented in

    Skafte, Nicki, Martin Jørgensen, and Søren Hauberg. 
    "Reliable training and estimation of variance networks." 
    Advances in Neural Information Processing Systems 32 (2019).

  Args:
      x (np.ndarray): [B] or [B x 1]
      scale (float, optional): Y-scale of function. Defaults to 1.0.

  Returns:
      Tuple[np.ndarray, np.ndarray]: Means and covarianes at x. [B x 1], [B x 1 x 1]
  """

  def f(x):
    return x * np.sin(x)

  variance = np.square(scale * (1 + x))

  return (scale * f(x)).reshape(-1, 1), variance.reshape(-1, 1, 1)


SF_FUNCTION_CONFIG = {
  'constant': (lambda x: sample_wrapper(sf_constant, x), (-1.0, 1.0), 2000),
  'n_constant': (lambda x: sample_wrapper(
    lambda x: sf_n_constant(x, x_range=[-1.0, 1.0], y_range=[-3.0, 3.0], noises=[0.01, 0.1, 1.0, 10.0]), x),
                 (-1.0, 1.0), 2000),
  'pitfalls': (lambda x: sample_wrapper(sf_pitfalls_sinusoidal, x), (0.0, 12.0), 2000),
  'detlefsen': (lambda x: sample_wrapper(sf_detlefsen_sinusoidal, x), (0.0, 10.0), 2000)
}


###############################################################################
### 3D ########################################################################
def spiral_3d(t, R=0.5, a=1 / np.pi):
  x = R * np.cos(t)
  y = R * np.sin(t)
  z = a * t
  return np.stack([x, y, z]).T


def corkscrew_3d(t, R=0.5, alpha=np.pi / 2):
  x = R * np.cos(t)
  y = R * np.sin(t)
  z = R * t * np.tan(alpha)
  return x, y, z


def generate_covariance(rotations, eigenvalues):
  R = utils.rotation_matrix_around_standard_spans(rotations)
  return R @ np.diag(eigenvalues) @ R.T


def cf_function1(t):
  mean = spiral_3d(t, R=0.5, a=0.1)

  a = (np.sin(t) + 1.0 + 1e-4) / 8
  rotations = np.stack([t, np.full_like(t, fill_value=np.pi / 3)]).T
  eigenvalues = np.stack([np.full_like(t, fill_value=0.1), np.full_like(t, fill_value=0.2), a]).T
  covariance = np.asarray(
    [generate_covariance(rotations_, eigenvalues_) for rotations_, eigenvalues_ in zip(rotations, eigenvalues)])

  return mean, covariance @ (0.01 * np.eye(3))

def cf_function2(t, cov_scale=1e-2):
  mean = spiral_3d(t, R=0.5, a=0.05)
  
  # this is a sinusoidal function with flat peaks and valleys, depending on b
  f = lambda x, b : np.sqrt((1 + b**2) / (1 + b**2 * np.cos(x)**2)) * np.cos(x)

  # a = 0.1 * 0.5 * (f(t, 10) + 1.0 + 1e-4)
  a = -0.01 * np.square(t - 10.0) + 0.1
  a = np.clip(a, a_min=1e-4, a_max=None)
  rotations = np.stack([t, np.full_like(t, fill_value=np.pi / 3)]).T
  eigenvalues = np.stack([np.full_like(t, fill_value=0.1), np.full_like(t, fill_value=0.2), a]).T
  covariance = np.asarray(
    [generate_covariance(rotations_, eigenvalues_) for rotations_, eigenvalues_ in zip(rotations, eigenvalues)])

  return mean, covariance @ (cov_scale * np.eye(3))