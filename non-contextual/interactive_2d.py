from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from utils.plotting import plot_diag_confidence_ellipse, plot_arrow
from utils.common import multivariate_normal_nll, multivariate_normal_nll_gradient, unit_vector



def plot_(mean, cov, new_mean, new_cov, dxmu, ax):
  # plot confidence ellipses
  plot_diag_confidence_ellipse(mean, cov, ax, edgecolor='red', linewidth=1.0, alpha=0.8, label='$\Sigma_{t-1}$')
  plot_diag_confidence_ellipse(new_mean, new_cov, ax, edgecolor='blue', linewidth=1.0, alpha=0.8, label='$\Sigma_{t}$')

  # plot covariance eigenvectors
  cov_vals, cov_vecs = np.linalg.eig(cov)
  plot_arrow(np.zeros(2), cov_vecs[:, 0], ax, c='red', linestyle=':', linewidth=3.0, alpha=0.5)
  plot_arrow(np.zeros(2), cov_vecs[:, 1], ax, c='red', linestyle=':', linewidth=3.0, alpha=0.5)
  ax.text(*(cov_vecs[:, 0] / 2), '$\lambda_1^{-1} x^*_1$', color='red', fontsize=8, alpha=0.5)
  
  plot_arrow(np.zeros(2), (1 / cov_vals[0]) * cov_vecs[:, 0], ax, c='red', linestyle=':', linewidth=1.0, alpha=0.5)
  plot_arrow(np.zeros(2), (1 / cov_vals[1]) * cov_vecs[:, 1], ax, c='red', linestyle=':', linewidth=1.0, alpha=0.5)
  ax.text(*(cov_vecs[:, 1] / 2), '$\lambda_2^{-1} x^*_2$', color='red', fontsize=8, alpha=0.5)
  
  new_cov_vals, new_cov_vecs = np.linalg.eig(new_cov)
  plot_arrow(np.zeros(2), new_cov_vecs[:, 0], ax, c='blue', linestyle=':', linewidth=3.0, alpha=0.5)
  plot_arrow(np.zeros(2), new_cov_vecs[:, 1], ax, c='blue', linestyle=':', linewidth=3.0, alpha=0.5)
  plot_arrow(np.zeros(2), (1 / new_cov_vals[0]) * new_cov_vecs[:, 0], ax, c='blue', linestyle=':', linewidth=1.0, alpha=0.5)
  plot_arrow(np.zeros(2), (1 / new_cov_vals[1]) * new_cov_vecs[:, 1], ax, c='blue', linestyle=':', linewidth=1.0, alpha=0.5)


  ## calculate angle between dxmu and both eigenvectors  
  cos_phi0 = np.dot(unit_vector(dxmu), unit_vector(cov_vecs[:, 0]))
  cos_phi1 = np.dot(unit_vector(dxmu), unit_vector(cov_vecs[:, 1]))

  phi = np.arccos([cos_phi0, cos_phi1])
  phi[phi > np.pi / 2] -= np.pi
  #phi = np.pi / 2 - phi

  to_eigv = phi[1] * (1 / cov_vals[0]) * cov_vecs[:, 0] + phi[0] * (1 / cov_vals[1]) * cov_vecs[:, 1]
  
  plot_arrow(np.zeros(2), to_eigv, ax, c='orange', linestyle='--', linewidth=3.0, alpha=0.5)
  

  # plot dxmu
  x, y = np.stack([np.zeros((2)), dxmu], axis=1)
  ax.plot(x, y, color='green', label='x - mu')
  
  ax.legend()



axis_color = 'lightgoldenrodyellow'

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.15, bottom=0.15)



mean_0 = np.zeros((2))
cov_0 = np.asarray([[1.0, 0.0], [0.0, 1.0]])
rot_0 = 45.0

def rotation_matrix_2d(deg):
  theta = np.deg2rad(deg)
  c, s = np.cos(theta), np.sin(theta)
  return np.array(((c, -s), (s, c)))

alpha = 1e-2
dxmu1_0 = dxmu2_0 = 0.0
ev1_0 = ev2_0 = 1.0

# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
rot_slider_ax  = fig.add_axes([0.15, 0.10, 0.65, 0.01], facecolor=axis_color)
rot_slider = Slider(rot_slider_ax, 'Rot', 0.0, 180.0, valinit=rot_0)

eigval1_slider_ax = fig.add_axes([0.15, 0.08, 0.65, 0.01], facecolor=axis_color)
eigval1_slider = Slider(eigval1_slider_ax, 'EV 1', 0.01, 1.9, valinit=ev1_0)

eigval2_slider_ax = fig.add_axes([0.15, 0.06, 0.65, 0.01], facecolor=axis_color)
eigval2_slider = Slider(eigval2_slider_ax, 'EV 2', 0.01, 1.9, valinit=ev2_0)

dxmu1_slider_ax = fig.add_axes([0.15, 0.04, 0.65, 0.01], facecolor=axis_color)
dxmu1_slider = Slider(dxmu1_slider_ax, 'DX 1', -50.0, 50.0, valinit=dxmu1_0)

dxmu2_slider_ax = fig.add_axes([0.15, 0.02, 0.65, 0.01], facecolor=axis_color)
dxmu2_slider = Slider(dxmu2_slider_ax, 'DX 2', -50.0, 50.0, valinit=dxmu2_0)


# Define an action for modifying the line when any slider's value changes
def on_slider_changed(val):
  ax.clear()

  # calculate new cov, new cov pair
  cov_ = np.diag([eigval1_slider.val, eigval2_slider.val])
  cov = rotation_matrix_2d(rot_slider.val) @ cov_ @ rotation_matrix_2d(rot_slider.val).T

  dxmu_ = np.asarray([dxmu1_slider.val, dxmu2_slider.val])

  _, cov_grad = multivariate_normal_nll_gradient(np.zeros((2)), dxmu_, cov)

  new_cov = cov - alpha * cov_grad

  # plot new stuff
  plot_(mean_0, cov, mean_0, new_cov, dxmu_, ax)
  
  ax.set_xlim([-2, 2])
  ax.set_ylim([-2, 2])

  ax.set_aspect('equal')
  
  fig.canvas.draw_idle()

rot_slider.on_changed(on_slider_changed)
eigval1_slider.on_changed(on_slider_changed)
eigval2_slider.on_changed(on_slider_changed)
dxmu1_slider.on_changed(on_slider_changed)
dxmu2_slider.on_changed(on_slider_changed)

plt.show()