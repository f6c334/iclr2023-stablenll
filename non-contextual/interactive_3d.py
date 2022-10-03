from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from utils.plotting import plot_diag_confidence_ellipse, plot_arrow
from utils.common import multivariate_normal_nll, multivariate_normal_nll_gradient, rotation_matrix_3d_x, rotation_matrix_3d_y





def rotation_matrix_3d(theta, psi):
  return rotation_matrix_3d_x(np.deg2rad(theta)) @ rotation_matrix_3d_y(np.deg2rad(psi))

def plot_confidence_ellipsoid(mean, cov, ax, **kwargs):
  u = np.linspace(0, 2 * np.pi, 33)
  v = np.linspace(0, np.pi, 33)

  x = np.outer(np.cos(u), np.sin(v))
  y = np.outer(np.sin(u), np.sin(v))
  z = np.outer(np.ones_like(u), np.cos(v))

  ellipsoid = (cov @ np.stack((x, y, z), 0).reshape(3, -1) + mean).reshape(3, *x.shape)

  return ax.plot_wireframe(*ellipsoid, rstride=4, cstride=4, alpha=0.5, **kwargs)

def plot_eigen(mean, cov, ax, **kwargs):
  cov_vals, cov_vecs = np.linalg.eig(cov)
  plot_arrow(np.zeros(3), cov_vecs[:, 0], ax, linestyle=':', linewidth=3.0, alpha=0.5, **kwargs)
  plot_arrow(np.zeros(3), cov_vecs[:, 1], ax, linestyle=':', linewidth=3.0, alpha=0.5, **kwargs)
  plot_arrow(np.zeros(3), cov_vecs[:, 2], ax, linestyle=':', linewidth=3.0, alpha=0.5, **kwargs)
  
  plot_arrow(np.zeros(3), (1 / cov_vals[0]) * cov_vecs[:, 0], ax, linestyle=':', linewidth=1.0, alpha=0.5, **kwargs)
  plot_arrow(np.zeros(3), (1 / cov_vals[1]) * cov_vecs[:, 1], ax, linestyle=':', linewidth=1.0, alpha=0.5, **kwargs)
  plot_arrow(np.zeros(3), (1 / cov_vals[2]) * cov_vecs[:, 2], ax, linestyle=':', linewidth=1.0, alpha=0.5, **kwargs)
  
  ax.text(*(cov_vecs[:, 0] / 2), '$\lambda_1^{-1} x^*_1$', fontsize=8, alpha=0.5, **kwargs)
  ax.text(*(cov_vecs[:, 1] / 2), '$\lambda_2^{-1} x^*_2$', fontsize=8, alpha=0.5, **kwargs)
  ax.text(*(cov_vecs[:, 2] / 2), '$\lambda_3^{-1} x^*_3$', fontsize=8, alpha=0.5, **kwargs)
  


mean = np.asarray([0.0, 0.0, 0.0]).reshape((3, 1))
axis_color = 'lightgoldenrodyellow'
phi_0 = psi_0 = 0.0
ev1_0 = ev2_0 = ev3_0 = 1.0
dxmu1_0 = dxmu2_0 = dxmu3_0 = 0.0
alpha = 1e-2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=0.15, bottom=0.25)


phi_slider_ax  = fig.add_axes([0.15, 0.22, 0.65, 0.01], facecolor=axis_color)
phi_slider = Slider(phi_slider_ax, 'Phi', 0.0, 180.0, valinit=phi_0)

psi_slider_ax  = fig.add_axes([0.15, 0.20, 0.65, 0.01], facecolor=axis_color)
psi_slider = Slider(psi_slider_ax, 'Psi', 0.0, 180.0, valinit=psi_0)

eigval1_slider_ax = fig.add_axes([0.15, 0.18, 0.65, 0.01], facecolor=axis_color)
eigval1_slider = Slider(eigval1_slider_ax, 'EV 1', 0.01, 1.9, valinit=ev1_0)

eigval2_slider_ax = fig.add_axes([0.15, 0.16, 0.65, 0.01], facecolor=axis_color)
eigval2_slider = Slider(eigval2_slider_ax, 'EV 2', 0.01, 1.9, valinit=ev2_0)

eigval3_slider_ax = fig.add_axes([0.15, 0.14, 0.65, 0.01], facecolor=axis_color)
eigval3_slider = Slider(eigval3_slider_ax, 'EV 3', 0.01, 1.9, valinit=ev3_0)

dxmu1_slider_ax = fig.add_axes([0.15, 0.12, 0.65, 0.01], facecolor=axis_color)
dxmu1_slider = Slider(dxmu1_slider_ax, 'DX 1', -50.0, 50.0, valinit=dxmu1_0)

dxmu2_slider_ax = fig.add_axes([0.15, 0.10, 0.65, 0.01], facecolor=axis_color)
dxmu2_slider = Slider(dxmu2_slider_ax, 'DX 2', -50.0, 50.0, valinit=dxmu2_0)

dxmu3_slider_ax = fig.add_axes([0.15, 0.08, 0.65, 0.01], facecolor=axis_color)
dxmu3_slider = Slider(dxmu3_slider_ax, 'DX 3', -50.0, 50.0, valinit=dxmu3_0)


def on_slider_changed(val):
  ax.clear()

  # calculate new cov, new cov pair
  cov_ = np.diag([eigval1_slider.val, eigval2_slider.val, eigval3_slider.val])
  cov = rotation_matrix_3d(phi_slider.val, psi_slider.val) @ cov_ @ rotation_matrix_3d(phi_slider.val, psi_slider.val).T

  dxmu_ = np.asarray([dxmu1_slider.val, dxmu2_slider.val, dxmu3_slider.val])

  _, cov_grad = multivariate_normal_nll_gradient(np.zeros((3)), dxmu_, cov)

  new_cov = cov - alpha * cov_grad

  # plot new stuff
  plot_confidence_ellipsoid(mean, cov, ax, color='red', label='$\Sigma_{t-1}$')
  plot_eigen(mean, cov, ax, color='red')

  plot_confidence_ellipsoid(mean, new_cov, ax, color='blue', label='$\Sigma_{t}$')
  plot_eigen(mean, new_cov, ax, color='blue')
  
  plot_arrow(np.zeros((3)), dxmu_, ax, color='green', label='$x - \mu$')

  ax.set_xlim([-2, 2])
  ax.set_ylim([-2, 2])
  ax.set_zlim([-2, 2])
  
  ax.legend()
  
  fig.canvas.draw_idle()
  


phi_slider.on_changed(on_slider_changed)
psi_slider.on_changed(on_slider_changed)
eigval1_slider.on_changed(on_slider_changed)
eigval2_slider.on_changed(on_slider_changed)
eigval3_slider.on_changed(on_slider_changed)
dxmu1_slider.on_changed(on_slider_changed)
dxmu2_slider.on_changed(on_slider_changed)
dxmu3_slider.on_changed(on_slider_changed)




plt.show()