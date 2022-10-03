import os
import re

from cProfile import label
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt

from utils.common import multivariate_normal_nll, make_subfolders
from utils.plotting import plot_3d_func


### SIMPLE 1D GAUSSIAN LOGLIK DERIVATIVES
def gaussian_loglik(x, mu, var):
  return - 0.5 * (np.log(2.0 * np.pi) + np.log(var) + np.square(x - mu) / var)

def dloglik_dmean(x, mu, var):
  return (x - mu) / var

def dloglik_dvar(x, mu, var):
  return - 0.5 * (1.0 / var - np.square(x - mu) / np.square(var))

def d2loglik_dmean2(x, mu, var):
  return - 1.0 / var

def d2loglik_dvar2(x, mu, var):
  return - 0.5 * (- 1.0 / np.square(var) + 2.0 * np.square(x - mu) / np.power(var, 3.0))

### PARAMETRIZATION DERIVATIVES
def dvar_dlogvar(x, mu, var):
  return var

def dvar_dsqrtvar(x, mu, var):
  return 2 * np.sqrt(var)

def dvar_dvarinv(x, mu, var):
  return - np.square(var)

def dvar_dsoftplus(x, mu, var): # softplus(p) = var
  return 1.0 / (1.0 + np.exp(- var))

def dvar_dlogsqrtvar(x, mu, var):
  return 2.0 * var

def tractable(x, mu, var, beta):
  return np.sqrt(var) * np.exp(2 * beta * var * dloglik_dvar(x, mu, var))
  
def dtractable(x, mu, var, beta):
  return np.sqrt(var) * (1.0 - np.exp(- beta * var * dloglik_dvar(x, mu, var)))


### PLOTTING
def plot_gradients_on_var_dxmu(folder, subfolder, dxmus, vars):
  descriptions = [
    # vanilla / adam gradients
    ('Vanilla Variance', r'$\frac{\partial LL}{\partial \sigma^2}$'),
    ('Vanilla LogVariance', r'$\frac{\partial LL}{\partial \log \sigma^2}$'),
    ('Vanilla SqrtVariance', r'$\frac{\partial LL}{\partial \sigma}$'),
    ('Vanilla InvVariance', r'$\frac{\partial LL}{\partial \sigma^{-2}}$'),
    ('Vanilla SoftmaxVariance', r'$\frac{\partial LL}{\partial softmaxinv(\sigma^2)}$'),
    ('Vanilla LogSqrtVariance', r'$\frac{\partial LL}{\partial \log \sigma}$'),

    # pitfalls 05 gradients
    ('Pitfalls beta=0.5 Variance', r'$\frac{\partial LL}{\partial \sigma^2}$'),
    ('Pitfalls beta=0.5 LogVariance', r'$\frac{\partial LL}{\partial \log \sigma^2}$'),
    ('Pitfalls beta=0.5 SqrtVariance', r'$\frac{\partial LL}{\partial \sigma}$'),
    ('Pitfalls beta=0.5 InvVariance', r'$\frac{\partial LL}{\partial \sigma^{-2}}$'),
    ('Pitfalls beta=0.5 SoftmaxVariance', r'$\frac{\partial LL}{\partial softmaxinv(\sigma^2)}$'),
    ('Pitfalls beta=0.5 LogSqrtVariance', r'$\frac{\partial LL}{\partial \log \sigma}$'),

    # pitfalls 10 gradients
    ('Pitfalls beta=1.0 Variance', r'$\frac{\partial LL}{\partial \sigma^2}$'),
    ('Pitfalls beta=1.0 LogVariance', r'$\frac{\partial LL}{\partial \log \sigma^2}$'),
    ('Pitfalls beta=1.0 SqrtVariance', r'$\frac{\partial LL}{\partial \sigma}$'),
    ('Pitfalls beta=1.0 InvVariance', r'$\frac{\partial LL}{\partial \sigma^{-2}}$'),
    ('Pitfalls beta=1.0 SoftmaxVariance', r'$\frac{\partial LL}{\partial softmaxinv(\sigma^2)}$'),
    ('Pitfalls beta=1.0 LogSqrtVariance', r'$\frac{\partial LL}{\partial \log \sigma}$'),

    # various gradients
    ('Hessian Variance', r'Gauss-Newton Gradient'),
    ('Tractable beta=0.01 SqrtVariance', r'Tractable Gradient beta=0.01'),
    ('Tractable beta=0.1 SqrtVariance', r'Tractable Gradient beta=0.1'),
  ]
  dfs = [
    # vanilla / adam gradients
    lambda x, mu, var : dloglik_dvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dsoftplus(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # pitfalls gradients
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dsoftplus(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # pitfalls gradients
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dsoftplus(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # various gradients
    lambda x, mu, var : dloglik_dvar(x, mu, var) / d2loglik_dvar2(x, mu, var),
    lambda x, mu, var : dtractable(x, mu, var, 0.01),
    lambda x, mu, var : dtractable(x, mu, var, 0.1),
  ]

  plt.rcParams.update({'font.size': 12})

  def plot_gradient(title, xlabel, ylabel, df):
    fig, ax = plt.subplots()
    
    for dxmu in dxmus:
      x, mu = dxmu, 0.0
      ax.plot(vars, df(x, mu, vars), ':', alpha=1.0, label=r'$x - \mu = $' + str(dxmu))

    ax.set_yscale('symlog')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_title(title)

    ax.legend()
    fig.set_tight_layout(True)

  make_subfolders(folder, [subfolder])

  for (name, grad_name), df in zip(descriptions, dfs):
    plot_gradient(name, r'$\sigma^2$', grad_name, df)
    plt.savefig(os.path.join(folder, subfolder, f'{re.sub(r"[ .=]", "_", name.lower())}.svg'), dpi=1200)



def plot_newvar_on_var_dxmu(folder, subfolder, dxmus, vars):
  descriptions = [
    # vanilla / adam
    'Vanilla Variance',
    'Vanilla LogVariance',
    'Vanilla SqrtVariance',
    'Vanilla InvVariance',
    'Vanilla LogSqrtVariance',

    # pitfalls 05
    'Pitfalls beta_05 Variance',
    'Pitfalls beta_05 LogVariance',
    'Pitfalls beta_05 SqrtVariance',
    'Pitfalls beta_05 InvVariance',
    'Pitfalls beta_05 LogSqrtVariance',

    # pitfalls 10
    'Pitfalls beta_10 Variance',
    'Pitfalls beta_10 LogVariance',
    'Pitfalls beta_10 SqrtVariance',
    'Pitfalls beta_10 InvVariance',
    'Pitfalls beta_10 LogSqrtVariance',

    # various
    'Hessian Variance',
    'Tractable beta_001 SqrtVariance',
    'Tractable beta_01 SqrtVariance',
  ]
  newvar_funcs = [
    # vanilla / adam new var
    lambda x, mu, var : var + dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # pitfalls 05 newvar
    lambda x, mu, var : var + var ** 0.5 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # pitfalls 10 newvar
    lambda x, mu, var : var + var ** 1.0 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # various newvar
    lambda x, mu, var : var + dloglik_dvar(x, mu, var) / d2loglik_dvar2(x, mu, var),
    lambda x, mu, var : np.square(tractable(x, mu, var, 0.01)),
    lambda x, mu, var : np.square(tractable(x, mu, var, 0.1)),
  ]


  plt.rcParams.update({'font.size': 12})

  def plot_newvar(title, newvar_func):
    fig, ax = plt.subplots()

    for dxmu in dxmus:
      x, mu = dxmu, 0.0
      ax.plot(vars, newvar_func(x, mu, vars), ':', alpha=1.0, label=r'$x - \mu = $' + str(dxmu))

      ax.set_yscale('symlog')

      ax.set_xlabel(r'$\sigma^2$')
      ax.set_ylabel(r'$\sigma^2_{new}$')

      ax.set_title(name + r'$(\alpha = 1.0)$')
      
      ax.legend()
      fig.set_tight_layout(True)

  make_subfolders(folder, [subfolder])

  for name, newvar_func in zip(descriptions, newvar_funcs):
    plot_newvar(name, newvar_func)
    plt.savefig(os.path.join(folder, subfolder, f'{re.sub(r"[ .=]", "_", name.lower())}.svg'), dpi=1200)
  

def plot_1d_gauss_nll():
  descriptions = [
    # vanilla / adam gradients
    ('Vanilla Variance'),
    #('Vanilla LogVariance'),
    # ('Vanilla SqrtVariance'),
    # ('Vanilla InvVariance'),
    # ('Vanilla SoftmaxVariance'),
    # ('Vanilla LogSqrtVariance'),
  ]
  dfs = [
    # vanilla / adam
    (lambda x, mu, var : multivariate_normal_nll(x, mu, var), lambda x : x),
    #(lambda x, mu, var : multivariate_normal_nll(x, mu, np.exp(var)), lambda x : np.log(x)),
    # lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    # lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    # lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    # lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dsoftplus(x, mu, var),
    # lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),
  ]

  def plot_loss(title, nll_func, p_func):
    x_range = np.asarray((-10, 10))
    y_range = np.asarray((1e-6, 10))
    mesh_size = 50

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    func = lambda x, y : nll_func(0.0, np.reshape(x, (1, 1)), np.reshape(y, (1, 1, 1)))
    plot_3d_func(func, x_range, p_func(y_range), mesh_size, ax)

    #ax.set_zscale('log')
    
    ax.set_xlabel("$\delta_{x, \mu}$")
    ax.set_ylabel("$\sigma^2$")
    ax.set_zlabel("NLL")
    
    ax.set_title(title)

    fig.set_tight_layout(True)
  
  for desc, (func, p_func) in zip(descriptions, dfs):
    plot_loss(desc, func, p_func)
    plt.show()

    
def plot_1dgauss_newvar_3d(folder, subfolder, dxmu_range, var_range, mesh_size):
  descriptions = [
    # vanilla / adam
    'Vanilla Variance',
    'Vanilla LogVariance',
    'Vanilla SqrtVariance',
    'Vanilla InvVariance',
    'Vanilla LogSqrtVariance',

    # pitfalls 05
    'Pitfalls beta_05 Variance',
    'Pitfalls beta_05 LogVariance',
    'Pitfalls beta_05 SqrtVariance',
    'Pitfalls beta_05 InvVariance',
    'Pitfalls beta_05 LogSqrtVariance',

    # pitfalls 10
    'Pitfalls beta_10 Variance',
    'Pitfalls beta_10 LogVariance',
    'Pitfalls beta_10 SqrtVariance',
    'Pitfalls beta_10 InvVariance',
    'Pitfalls beta_10 LogSqrtVariance',

    # various
    'Hessian Variance',
    'Tractable beta_001 SqrtVariance',
    'Tractable beta_01 SqrtVariance',
  ]
  dfs = [
    # vanilla / adam new var
    lambda x, mu, var : var + dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # pitfalls 05 newvar
    lambda x, mu, var : var + var ** 0.5 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # pitfalls 10 newvar
    lambda x, mu, var : var + var ** 1.0 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var * np.exp(var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var)),
    lambda x, mu, var : np.square(np.sqrt(var) + var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var)),
    lambda x, mu, var : 1.0 / (1.0 / var + var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var)),
    lambda x, mu, var : var * np.exp(var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var)),

    # various newvar
    lambda x, mu, var : var + dloglik_dvar(x, mu, var) / d2loglik_dvar2(x, mu, var),
    lambda x, mu, var : np.square(tractable(x, mu, var, 0.01)),
    lambda x, mu, var : np.square(tractable(x, mu, var, 0.1)),
  ]
  

  def plot_loss(title, newvar_func):
    Z_MIN, Z_MAX = -1e2, 1e2
    x_range, y_range = dxmu_range, var_range

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    func = lambda x, y : newvar_func(0.0, x, y) if Z_MIN <= newvar_func(0.0, x, y) <= Z_MAX else np.nan
    plot_3d_func(func, x_range, y_range, mesh_size, ax, cmap='jet')

    #ax.set_zscale('symlog')
    
    ax.set_xlabel("$\delta_{x, \mu}$")
    ax.set_ylabel("$\sigma^2$")
    ax.set_zlabel("$\sigma^2_{new}$")
    
    ax.set_title(title)

    fig.set_tight_layout(True)
  
  make_subfolders(folder, [subfolder])

  for desc, func in zip(descriptions, dfs):
    plot_loss(desc, func)
    plt.savefig(os.path.join(folder, subfolder, f'{desc.lower().replace(" ", "_")}.svg'), dpi=1200)


def plot_1dgauss_gradient_3d(folder, subfolder, dxmu_range, var_range, mesh_size):
  descriptions = [
    # vanilla / adam
    'Vanilla Variance',
    'Vanilla LogVariance',
    'Vanilla SqrtVariance',
    'Vanilla InvVariance',
    'Vanilla LogSqrtVariance',

    # pitfalls 05
    'Pitfalls beta_05 Variance',
    'Pitfalls beta_05 LogVariance',
    'Pitfalls beta_05 SqrtVariance',
    'Pitfalls beta_05 InvVariance',
    'Pitfalls beta_05 LogSqrtVariance',

    # pitfalls 10
    'Pitfalls beta_10 Variance',
    'Pitfalls beta_10 LogVariance',
    'Pitfalls beta_10 SqrtVariance',
    'Pitfalls beta_10 InvVariance',
    'Pitfalls beta_10 LogSqrtVariance',

    # various
    'Hessian Variance',
    'Tractable beta_001 SqrtVariance',
    'Tractable beta_01 SqrtVariance',
  ]
  dfs = [
        # vanilla / adam gradients
    lambda x, mu, var : dloglik_dvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # pitfalls gradients
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : var ** 0.5 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # pitfalls gradients
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dsqrtvar(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dvarinv(x, mu, var),
    lambda x, mu, var : var ** 1.0 * dloglik_dvar(x, mu, var) * dvar_dlogsqrtvar(x, mu, var),

    # various gradients
    lambda x, mu, var : dloglik_dvar(x, mu, var) / d2loglik_dvar2(x, mu, var),
    lambda x, mu, var : dtractable(x, mu, var, 0.01),
    lambda x, mu, var : dtractable(x, mu, var, 0.1),
  ]
  

  def plot_loss(title, newvar_func):
    Z_MIN, Z_MAX = -1e2, 1e2
    x_range, y_range = dxmu_range, var_range

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    func = lambda x, y : newvar_func(0.0, x, y) if Z_MIN <= newvar_func(0.0, x, y) <= Z_MAX else np.nan
    plot_3d_func(func, x_range, y_range, mesh_size, ax, cmap='jet')

    #ax.set_zscale('symlog')
    
    ax.set_xlabel("$\delta_{x, \mu}$")
    ax.set_ylabel("$\sigma^2$")
    ax.set_zlabel("$\partial LL / \sigma^2$")
    
    ax.set_title(title)

    fig.set_tight_layout(True)
  
  make_subfolders(folder, [subfolder])

  for desc, func in zip(descriptions, dfs):
    plot_loss(desc, func)
    plt.savefig(os.path.join(folder, subfolder, f'{desc.lower().replace(" ", "_")}.svg'), dpi=1200)


folder = 'non-contextual/figures/1d_gradients_plots'


x_range = np.asarray((-1, 1))
y_range = np.asarray((1e-2, 1.0))
mesh_size = 1000

dxmus = [0.001, 0.01, 0.1, 1.0]
vars = np.linspace(0.01, 1.0, 10000)

plot_gradients_on_var_dxmu(folder, '1d_gradients_on_var_dxmu', dxmus, vars)
plt.close('all')
plot_newvar_on_var_dxmu(folder, '1d_newvar_on_var_dxmu', dxmus, vars)
plt.close('all')

# 3d gradient plots
plot_1dgauss_newvar_3d(folder, '3d_newvar_plots', x_range, y_range, mesh_size)
plt.close('all')
plot_1dgauss_gradient_3d(folder, '3d_gradient_plots', x_range, y_range, mesh_size)
plt.close('all')

x_range = np.asarray((-10, 10))
y_range = np.asarray((1.0, 10.0))

plot_1dgauss_newvar_3d(folder, '3d_newvar_plots_far', x_range, y_range, mesh_size)
plt.close('all')
plot_1dgauss_gradient_3d(folder, '3d_gradient_plots_far', x_range, y_range, mesh_size)
plt.close('all')