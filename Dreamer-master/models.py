"""DreamerV2 model components adapted from the official DreamerV2 design.

The world-model structure follows danijar/dreamerv2: categorical RSSM,
KL balancing, image/reward/discount heads, and imagined actor-critic learning.
The only task-specific extension is HybridActionDecoder for the UAV benchmark's
parameterized action space (categorical action + continuous per-action parameter).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras import mixed_precision as prec
import tools


class RSSM(tools.Module):
  """DreamerV2 categorical recurrent state-space model.

  State = deterministic GRU state + `stoch` categorical variables, each with
  `discrete` classes. Samples use the straight-through estimator.
  """

  def __init__(self, stoch=32, deter=400, hidden=400, discrete=32, act=tf.nn.elu):
    super().__init__()
    self._stoch = int(stoch)
    self._deter = int(deter)
    self._hidden = int(hidden)
    self._discrete = int(discrete)
    self._act = act
    self._cell = tfkl.GRUCell(self._deter)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    logits = tf.zeros([batch_size, self._stoch, self._discrete], dtype)
    stoch = tf.zeros_like(logits)
    deter = tf.zeros([batch_size, self._deter], dtype)
    return dict(logits=logits, stoch=stoch, deter=deter)

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    return {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in prior.items()}

  def get_feat(self, state):
    stoch = tf.reshape(state['stoch'], tf.concat([tf.shape(state['stoch'])[:-2], [-1]], 0))
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    return tfd.Independent(tfd.OneHotCategorical(logits=state['logits']), 1)

  def _sample(self, logits):
    # Straight-through categorical sample used by official DreamerV2.
    dist = tfd.OneHotCategorical(logits=logits)
    sample = tf.cast(dist.sample(), logits.dtype)
    probs = tf.nn.softmax(logits, -1)
    return sample + probs - tf.stop_gradient(probs)

  def _stats(self, x):
    logits = self.get('logits', tfkl.Dense, self._stoch * self._discrete)(x)
    logits = tf.reshape(logits, tf.concat([tf.shape(x)[:-1], [self._stoch, self._discrete]], 0))
    stoch = self._sample(logits)
    return dict(logits=logits, stoch=stoch)

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._stats(x)
    return dict(**stats, deter=prior['deter']), prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    stoch = tf.reshape(prev_state['stoch'], [tf.shape(prev_state['stoch'])[0], -1])
    x = tf.concat([stoch, prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden, self._act)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]
    x = self.get('img2', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._stats(x)
    return dict(**stats, deter=deter)

  def kl_loss(self, post, prior, balance=0.8, free=0.0, forward=False):
    """DreamerV2 KL balancing with stop-gradient on opposite sides."""
    lhs, rhs = (prior, post) if forward else (post, prior)
    lhs_sg = {k: tf.stop_gradient(v) for k, v in lhs.items()}
    rhs_sg = {k: tf.stop_gradient(v) for k, v in rhs.items()}
    value_lhs = tfd.kl_divergence(self.get_dist(lhs), self.get_dist(rhs_sg))
    value_rhs = tfd.kl_divergence(self.get_dist(lhs_sg), self.get_dist(rhs))
    loss_lhs = tf.maximum(tf.reduce_mean(value_lhs), float(free))
    loss_rhs = tf.maximum(tf.reduce_mean(value_rhs), float(free))
    loss = float(balance) * loss_lhs + (1.0 - float(balance)) * loss_rhs
    value = tf.reduce_mean(tfd.kl_divergence(self.get_dist(post), self.get_dist(prior)))
    return loss, value


class ConvEncoder(tools.Module):
  def __init__(self, depth=48, act=tf.nn.elu):
    self._act, self._depth = act, depth
  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', tfkl.Conv2D, 1*self._depth, 4, **kwargs)(x)
    x = self.get('h2', tfkl.Conv2D, 2*self._depth, 4, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2D, 4*self._depth, 4, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2D, 8*self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32*self._depth]], 0)
    return tf.reshape(x, shape)


class ConvDecoder(tools.Module):
  def __init__(self, depth=48, act=tf.nn.elu, shape=(64,64,3)):
    self._act, self._depth, self._shape = act, depth, shape
  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', tfkl.Dense, 32*self._depth)(features)
    x = tf.reshape(x, [-1,1,1,32*self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 4*self._depth, 5, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 2*self._depth, 5, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 1*self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseHead(tools.Module):
  def __init__(self, shape, layers=4, units=400, dist='mse', act=tf.nn.elu):
    self._shape, self._layers, self._units, self._dist, self._act = shape, layers, units, dist, act
  def __call__(self, features):
    x = features
    for i in range(self._layers):
      x = self.get(f'h{i}', tfkl.Dense, self._units, self._act)(x)
    x = self.get('out', tfkl.Dense, int(np.prod(self._shape) or 1))(x)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if self._dist == 'mse':
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(logits=x), len(self._shape))
    raise NotImplementedError(self._dist)


class HybridDist:
  """Stable parameterized-action distribution for UAV control.

  The discrete branch uses a straight-through categorical sample. The
  continuous branch uses a reparameterized Gaussian followed by tanh for
  environment/RSSM actions, but policy log-probabilities are evaluated in the
  pre-tanh Gaussian space. This avoids the singular tanh inverse/Jacobian near
  +/-1 that caused very large actor gradients.

  Only the parameter belonging to the selected discrete action contributes to
  the REINFORCE log-probability and entropy. Unselected parameter heads remain
  available for differentiable imagination but do not add irrelevant policy
  gradient terms.
  """
  def __init__(self, logits, mean, std):
    self.logits = logits
    self.mean_tensor = mean
    self.std_tensor = std
    self._cat = tfd.Categorical(logits=logits)
    self._normal = tfd.Normal(mean, std)

  def _onehot_st(self, index):
    # Hard non-reparameterized categorical sample. Discrete logits are trained
    # only by REINFORCE, not by a straight-through dynamics surrogate.
    hard = tf.one_hot(index, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    return tf.stop_gradient(hard)

  def sample(self):
    idx = self._cat.sample()
    select = self._onehot_st(idx)
    # Reparameterized continuous sample. Keep the pre-tanh value internally in
    # the graph and expose only bounded parameters to the RSSM/environment.
    raw = self.mean_tensor + self.std_tensor * tf.random.normal(
        tf.shape(self.mean_tensor), dtype=self.mean_tensor.dtype)
    params = tf.tanh(raw)
    return tf.concat([select, params], -1)

  def mode(self):
    idx = tf.argmax(self.logits, -1, output_type=tf.int32)
    select = tf.one_hot(idx, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    params = tf.tanh(self.mean_tensor)
    return tf.concat([select, params], -1)

  def entropy(self):
    # Entropy regularizes the categorical choice and only the parameter of the
    # currently most likely action. This avoids summing K irrelevant parameter
    # entropies for a single parameterized action.
    idx = tf.argmax(self.logits, -1, output_type=tf.int32)
    onehot = tf.one_hot(idx, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    param_ent = tf.reduce_sum(self._normal.entropy() * onehot, -1)
    return self._cat.entropy() + param_ent

  def discrete_log_prob(self, action):
    """Log-probability of only the discrete action branch."""
    k = tf.shape(self.logits)[-1]
    select = action[..., :k]
    idx = tf.argmax(tf.stop_gradient(select), -1, output_type=tf.int32)
    return self._cat.log_prob(idx)

  def log_prob(self, action):
    k = tf.shape(self.logits)[-1]
    select, params = action[..., :k], action[..., k:]
    idx = tf.argmax(select, -1, output_type=tf.int32)
    onehot = tf.one_hot(idx, k, dtype=params.dtype)

    # Stable pre-tanh approximation. Stop gradients through the sampled action
    # value for the score-function term; gradients flow through distribution
    # parameters only, as required by REINFORCE.
    safe = tf.clip_by_value(tf.stop_gradient(params), -0.999, 0.999)
    raw = tf.atanh(safe)
    param_logprob = tf.reduce_sum(self._normal.log_prob(raw) * onehot, -1)
    return self._cat.log_prob(idx) + param_logprob


class HybridActionDecoder(tools.Module):
  """DreamerV2 actor extended to categorical + action-conditioned parameter."""
  def __init__(self, num_actions, layers=4, units=400, min_std=0.1, act=tf.nn.elu):
    self._num_actions = int(num_actions)
    self._layers, self._units, self._min_std, self._act = layers, units, min_std, act

  def __call__(self, features):
    x = features
    for i in range(self._layers):
      x = self.get(f'h{i}', tfkl.Dense, self._units, self._act)(x)
    logits = self.get('logits', tfkl.Dense, self._num_actions)(x)
    stats = self.get('params', tfkl.Dense, 2 * self._num_actions)(x)
    mean, raw_std = tf.split(stats, 2, -1)

    # Bound the Gaussian location and scale. The previous unbounded mean/std
    # could rapidly saturate tanh and create extreme score-function gradients.
    mean = 2.0 * tf.tanh(mean / 2.0)
    std = tf.nn.softplus(raw_std)
    std = tf.clip_by_value(std, self._min_std, 1.5)
    return HybridDist(logits, mean, std)
