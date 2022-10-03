import tensorflow as tf
import tensorflow_probability as tfp


class StandardGaussianMLP(tf.keras.Model):

  def __init__(self, input_shape, gauss_dimension=2, hidden_layers=[50, 50], activations='relu', **kwargs) -> None:
    super(StandardGaussianMLP, self).__init__(**kwargs)

    self.gauss_dimension = gauss_dimension
    self.activations = activations

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers = [tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers]

    self.mean_layer = tf.keras.layers.Dense(gauss_dimension, name='mean')
    self.covariance_layer = tf.keras.layers.Dense(gauss_dimension * (gauss_dimension + 1) / 2, name='covariance')

    self.add_layer = tf.keras.layers.Add()

  def call(self, inputs):
    x = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers:
      x = hidden_layer(x)

    mean = self.mean_layer(x)
    covariance_cholesky = tfp.math.fill_triangular(self.covariance_layer(x))
    # covariance_cholesky += 1e-8 * tf.eye(num_rows=tf.shape(covariance_cholesky)[-1], dtype=covariance_cholesky.dtype)

    return mean, covariance_cholesky


class StandardGaussianSplitMLP(tf.keras.Model):

  def __init__(self, input_shape, gauss_dimension=2, hidden_layers=[50, 50], activations='relu', **kwargs) -> None:
    super(StandardGaussianSplitMLP, self).__init__(**kwargs)

    self.gauss_dimension = gauss_dimension
    self.activations = activations

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers_mean = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]
    self.hidden_layers_covariance = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]

    self.mean_layer = tf.keras.layers.Dense(gauss_dimension, name='mean')
    self.covariance_layer = tf.keras.layers.Dense(gauss_dimension * (gauss_dimension + 1) / 2, name='covariance')

  def call(self, inputs):
    x = x_ = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers_mean:
      x = hidden_layer(x)

    for hidden_layer in self.hidden_layers_covariance:
      x_ = hidden_layer(x_)

    mean = self.mean_layer(x)
    covariance_cholesky = tfp.math.fill_triangular(self.covariance_layer(x_))

    return mean, covariance_cholesky


class UnivariateGaussianSplitMLP(tf.keras.Model):

  def __init__(self, input_shape, gauss_dimension=2, hidden_layers=[50, 50], activations='relu', covdiag_min=1e-3, covdiag_max=1e3, **kwargs) -> None:
    super(UnivariateGaussianSplitMLP, self).__init__(**kwargs)

    self.gauss_dimension = gauss_dimension
    self.activations = activations
    self.covdiag_min = covdiag_min
    self.covdiag_max = covdiag_max

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers_mean = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]
    self.hidden_layers_covariance = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]

    self.mean_layer = tf.keras.layers.Dense(gauss_dimension, name='mean')
    self.covariance_layer = tf.keras.layers.Dense(gauss_dimension, name='covariance')

  def call(self, inputs):
    x = x_ = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers_mean:
      x = hidden_layer(x)

    for hidden_layer in self.hidden_layers_covariance:
      x_ = hidden_layer(x_)

    mean = self.mean_layer(x)
    covariance_cholesky = self.covariance_layer(x_)

    return mean, covariance_cholesky


class GaussianMLP(tf.keras.Model):
  """ Split mean and covariance head """

  def __init__(self,
               input_shape,
               n_dims=1,
               gauss_dimension=1,
               hidden_layers=[50, 50],
               activations='relu',
               **kwargs) -> None:
    super(GaussianMLP, self).__init__(**kwargs)

    self.n_dims = n_dims
    self.gauss_dimension = gauss_dimension
    self.gauss_covariance_dimension = int(gauss_dimension * (gauss_dimension + 1) / 2)
    self.activations = activations

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers = [tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers]

    self.mean_layer = tf.keras.layers.Dense(n_dims * gauss_dimension, name='mean')
    self.mean_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension))

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension, name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, self.gauss_covariance_dimension))

  def call(self, inputs):
    x = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers:
      x = hidden_layer(x)

    mean = self.mean_reshape(self.mean_layer(x))
    covariance_cholesky = self.covariance_reshape(self.covariance_layer(x))
    """ test """
    covariance_cholesky = tfp.math.fill_triangular(covariance_cholesky)
    covariance = tf.linalg.matmul(covariance_cholesky, covariance_cholesky, transpose_b=True)

    min_var, max_var = 1e-8, 100
    covariance = tf.clip_by_value(covariance, clip_value_min=min_var, clip_value_max=max_var)

    return mean, tf.linalg.cholesky(covariance)

    # min_var, max_var = 1e-8, 100  # TODO: add to init and correct if sign == 0
    # # covariance_cholesky = tf.where(covariance_cholesky )
    # covariance_cholesky = tf.math.sign(covariance_cholesky) * tf.clip_by_value(covariance_cholesky, clip_value_min=tf.sqrt(min_var), clip_value_max=tf.sqrt(max_var))
    # # covariance_cholesky = tf.clip_by_value(covariance_cholesky, clip_value_min=tf.sqrt(min_var), clip_value_max=tf.sqrt(max_var))

    # return mean, tfp.math.fill_triangular(covariance_cholesky)


class DynamicSoftplus(tf.keras.layers.Layer):

  def __init__(self):
    super(DynamicSoftplus, self).__init__()

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1])])

  def call(self, inputs):
    return self.log(1.0 + self.exp(inputs, self.kernel), self.kernel)

  def log(self, x, b):
    return tf.math.log(x) / tf.math.log(b)

  def exp(self, x, b):
    return b**x


class DynamicReLU(tf.keras.layers.Layer):

  def __init__(self):
    super(DynamicSoftplus, self).__init__()

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1])])

  def call(self, inputs):
    return tf.keras.activations.relu(inputs + self.kernel)


###############################################################################
### MAIN ARCHITECTURE #########################################################
### 1D Parametrizations
class LogCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(LogCovarianceHead, self).__init__()

    assert gauss_dimension == 1, 'Gauss dimension for this head must be 1'
    self.covariance_layer = tf.keras.layers.Dense(n_dims, activation=covariance_activation, name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.math.exp(A)


class LogStdCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(LogStdCovarianceHead, self).__init__()

    assert gauss_dimension == 1, 'Gauss dimension for this head must be 1'
    self.covariance_layer = tf.keras.layers.Dense(n_dims, activation=covariance_activation, name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.math.square(tf.math.exp(A))


class SoftplusCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(SoftplusCovarianceHead, self).__init__()

    assert gauss_dimension == 1, 'Gauss dimension for this head must be 1'
    self.covariance_layer = tf.keras.layers.Dense(n_dims, activation=covariance_activation, name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.keras.activations.softplus(A)


class SoftplusStdCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(SoftplusStdCovarianceHead, self).__init__()

    self.covdiag_min, self.covdiag_max = cov_stab

    assert gauss_dimension == 1, 'Gauss dimension for this head must be 1'
    self.covariance_layer = tf.keras.layers.Dense(n_dims, activation=covariance_activation, name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = tf.keras.activations.softplus(self.covariance_layer(inputs))
    A = self.covariance_reshape(tf.clip_by_value(x, clip_value_min=self.covdiag_min, clip_value_max=self.covdiag_max))
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.math.square(A)


### ND Parametrizations
class CholeskyCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(CholeskyCovarianceHead, self).__init__()

    self.gauss_dimension = gauss_dimension
    self.gauss_covariance_dimension = int(gauss_dimension * (gauss_dimension + 1) / 2)
    self.covdiag_min, self.covdiag_max = cov_stab

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension,
                                                  activation=covariance_activation,
                                                  name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, self.gauss_covariance_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = tfp.math.fill_triangular(self.covariance_reshape(x))
    A_diag = tf.linalg.diag_part(A)
    A_diag = tf.where(A_diag >= 0, x=tf.clip_by_value(A_diag, clip_value_min=self.covdiag_min, clip_value_max=self.covdiag_max), y=tf.clip_by_value(A_diag, clip_value_min=-self.covdiag_max, clip_value_max=-self.covdiag_min))
    A = tf.linalg.set_diag(A, A_diag)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.linalg.matmul(A, A, transpose_b=True)


class SqrtCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(SqrtCovarianceHead, self).__init__()

    self.gauss_covariance_dimension = int(gauss_dimension * gauss_dimension)

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension,
                                                  activation=covariance_activation,
                                                  name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.linalg.matmul(A, A)

class SqrtTCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(SqrtTCovarianceHead, self).__init__()

    self.gauss_dimension = gauss_dimension
    self.gauss_covariance_dimension = int(gauss_dimension * gauss_dimension)

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension,
                                                  activation=covariance_activation,
                                                  name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    return tf.linalg.matmul(A, A, transpose_b=True)

class LogmCovarianceHead(tf.keras.Model):

  def __init__(self, n_dims, gauss_dimension, covariance_activation='linear', cov_stab=(1e-3, 1e3)):
    super(LogmCovarianceHead, self).__init__()

    self.gauss_covariance_dimension = int(gauss_dimension * gauss_dimension)

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension,
                                                  activation=covariance_activation,
                                                  name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension, gauss_dimension))

  def call(self, inputs):
    x = self.covariance_layer(inputs)
    A = self.covariance_reshape(x)
    return self.to_covariance(A), A

  def to_covariance(self, A):
    M = tf.linalg.matmul(A, A, transpose_b=True)
    return tf.linalg.expm(M)

_COVARIANCE_HEADS = {
  'sqrt': CholeskyCovarianceHead,
  'logvar': LogCovarianceHead,
  'logstd': LogStdCovarianceHead,
  'softplusvar': SoftplusCovarianceHead,
  'softplusstd': SoftplusStdCovarianceHead,
  'cholesky': CholeskyCovarianceHead,
  'sqrtm': SqrtCovarianceHead,
  'sqrtmT': SqrtTCovarianceHead,
  'logm': LogmCovarianceHead,
}


class GaussianSplitMLP(tf.keras.Model):
  """ Split mean and covariance head """

  def __init__(self,
               input_shape,
               n_dims=1,
               gauss_dimension=1,
               hidden_layers=[50, 50],
               activations='relu',
               covariance_activation='linear',
               covariance_head_type='cholesky',
               cov_stab=(1e-3, 1e3),
               **kwargs) -> None:
    super(GaussianSplitMLP, self).__init__(**kwargs)

    self.n_dims = n_dims
    self.gauss_dimension = gauss_dimension
    self.activations = activations

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers_mean = [
      tf.keras.layers.Dense(layer_size, activation=activations, name=f'mean_dense_{i}') for i, layer_size in enumerate(hidden_layers)
    ]
    self.hidden_layers_covariance = [
      tf.keras.layers.Dense(layer_size, activation=activations, name=f'covariance_dense_{i}') for i, layer_size in enumerate(hidden_layers)
    ]

    self.mean_layer = tf.keras.layers.Dense(n_dims * gauss_dimension, name='mean')
    self.mean_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension))

    self.covariance_head = _COVARIANCE_HEADS[covariance_head_type](self.n_dims,
                                                                   self.gauss_dimension,
                                                                   covariance_activation=covariance_activation,
                                                                   cov_stab=cov_stab)

  def call(self, inputs):
    x = x_ = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers_mean:
      x = hidden_layer(x)

    for hidden_layer in self.hidden_layers_covariance:
      x_ = hidden_layer(x_)

    mean = self.mean_reshape(self.mean_layer(x))
    covariance, covariance_parametrization = self.covariance_head(x_)
    # + tf.eye(self.gauss_dimension, dtype=covariance_cholesky.dtype)

    return mean, covariance, covariance_parametrization


###############################################################################
###############################################################################


class DynamicGaussianSplitMLP(tf.keras.Model):
  """ Split mean and covariance head """

  def __init__(self,
               input_shape,
               n_dims=1,
               gauss_dimension=1,
               hidden_layers=[50, 50],
               activations='relu',
               covariance_activation='linear',
               **kwargs) -> None:
    super(DynamicGaussianSplitMLP, self).__init__(**kwargs)

    self.n_dims = n_dims
    self.gauss_dimension = gauss_dimension
    self.gauss_covariance_dimension = int(gauss_dimension * (gauss_dimension + 1) / 2)
    self.activations = activations

    # standard mlp with hidden_layers
    self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name=None)
    self.hidden_layers_mean = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]
    self.hidden_layers_covariance = [
      tf.keras.layers.Dense(layer_size, activation=activations) for layer_size in hidden_layers
    ]

    self.mean_layer = tf.keras.layers.Dense(n_dims * gauss_dimension, name='mean')
    self.mean_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, gauss_dimension))

    self.covariance_layer = tf.keras.layers.Dense(n_dims * self.gauss_covariance_dimension,
                                                  activation=covariance_activation,
                                                  name='covariance')
    self.covariance_reshape = tf.keras.layers.Reshape(target_shape=(n_dims, self.gauss_covariance_dimension))
    self.dyn_softplus = DynamicSoftplus()

  def call(self, inputs):
    x = x_ = self.input_layer(inputs)

    for hidden_layer in self.hidden_layers_mean:
      x = hidden_layer(x)

    for hidden_layer in self.hidden_layers_covariance:
      x_ = hidden_layer(x_)

    mean = self.mean_reshape(self.mean_layer(x))
    covariance_cholesky = self.covariance_reshape(self.covariance_layer(x_))
    covariance_cholesky = self.dyn_softplus(covariance_cholesky)

    # covariance_cholesky = tf.keras.activations.softplus(covariance_cholesky) # TODO: remove, this is just temporary (FOR 1D)
    covariance_cholesky = tfp.math.fill_triangular(
      covariance_cholesky)  # + tf.eye(self.gauss_dimension, dtype=covariance_cholesky.dtype)

    return mean, covariance_cholesky
