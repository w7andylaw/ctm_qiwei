"""DreamerV2 for the UAV parameterized-action benchmark.

Core algorithm follows the official danijar/dreamerv2 design:
- categorical RSSM with straight-through samples
- KL balancing
- image, reward, and discount world-model heads
- latent imagination actor-critic
- mixed dynamics/REINFORCE actor gradient
- slow target critic

The only benchmark-specific extension is a hybrid actor distribution for the
paper's (discrete action, continuous parameter) action space. There is no
separate UAV adapter module; the environment contract lives in envs.py.
"""
from __future__ import annotations
import argparse, collections, functools, json, os, pathlib, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision as prec
import models
import tools
import wrappers


def define_config():
  c = tools.AttrDict()
  # Runtime / benchmark. Paper-comparison defaults use Task 2.
  c.logdir = pathlib.Path('./outputs/dreamerv2_uav_relay')
  c.seed = 0
  c.task = 'uav_relay'
  c.steps = 5e6
  c.eval_every = 1e4
  c.log_every = 1e3
  c.envs = 1
  c.parallel = 'process'
  c.action_repeat = 1
  c.time_limit = 100
  c.prefill = 5000
  c.precision = 32
  c.gpu_growth = True
  c.log_images = False
  c.smoke = False

  # Replay. Official V2 defaults are batch=50,length=50; length=20 is the
  # explicit UAV adaptation because paper episodes can terminate before 50.
  c.batch_size = 50
  c.batch_length = 10
  c.replay_capacity = 2_000_000
  c.dataset_prefetch = 2
  c.train_every = 5
  c.train_steps = 1
  c.pretrain = 100

  # Official DreamerV2 world-model structure/default scale.
  c.rssm_hidden = 400
  c.rssm_deter = 400
  c.rssm_stoch = 32
  c.rssm_discrete = 32
  c.cnn_depth = 48
  c.num_units = 400
  c.model_lr = 3e-4
  c.kl_scale = 1.0
  c.kl_balance = 0.8
  c.kl_free = 0.0
  c.pred_discount = True
  c.discount_scale = 1.0
  c.grad_clip = 100.0

  # Official V2 actor-critic structure.
  c.actor_lr = 1e-4
  c.critic_lr = 1e-4
  c.discount = 0.99
  c.discount_lambda = 0.95
  c.imag_horizon = 15
  # UAV hybrid actor: discrete branch uses REINFORCE; continuous parameters use dynamics gradients.
  c.parameter_grad_scale = 1.0
  c.actor_ent = 1e-4
  c.slow_target = True
  c.slow_target_update = 100
  c.slow_target_fraction = 1.0
  c.actor_min_std = 0.1
  return c


class DreamerV2(tools.Module):
  def __init__(self, config, datadir, actspace, writer):
    self.c = config
    self.writer = writer
    self.actdim = int(actspace.shape[0])
    if not str(config.task).startswith('uav_'):
      raise ValueError('This cleaned build is intentionally UAV-only.')
    self.num_actions = 3 if config.task == 'uav_relay' else 2
    if self.actdim != 2 * self.num_actions:
      raise ValueError('UAV action vector must be [K selections, K parameters].')
    with tf.device('/CPU:0'):
      self.step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self.should_train = tools.Every(config.train_every)
    self.should_log = tools.Every(config.log_every)
    self.should_pretrain = tools.Once()
    metric_names = (
        'model_loss', 'image_loss', 'reward_loss', 'discount_loss', 'kl',
        'actor_loss', 'critic_loss', 'model_grad_norm', 'actor_grad_norm',
        'critic_grad_norm')
    self.metrics = {
        name: tf.keras.metrics.Mean(name=name) for name in metric_names}
    self.float = prec.global_policy().compute_dtype
    self.dataset = iter(load_dataset(datadir, config))
    self._build_model()

  def _build_model(self):
    self.encoder = models.ConvEncoder(self.c.cnn_depth, tf.nn.elu)
    self.rssm = models.RSSM(
        self.c.rssm_stoch, self.c.rssm_deter, self.c.rssm_hidden,
        self.c.rssm_discrete, tf.nn.elu)
    self.decoder = models.ConvDecoder(self.c.cnn_depth, tf.nn.elu)
    self.reward = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.discount = models.DenseHead((), 4, self.c.num_units, 'binary', tf.nn.elu)
    self.actor = models.HybridActionDecoder(
        self.num_actions, 4, self.c.num_units, self.c.actor_min_std, tf.nn.elu)
    self.critic = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.slow_critic = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.model_opt = tools.Adam('model', [self.encoder,self.rssm,self.decoder,self.reward,self.discount],
                                self.c.model_lr, clip=self.c.grad_clip)
    self.actor_opt = tools.Adam('actor', [self.actor], self.c.actor_lr, clip=min(self.c.grad_clip, 20.0))
    self.critic_opt = tools.Adam('critic', [self.critic], self.c.critic_lr, clip=self.c.grad_clip)
    # Materialize variables and initialize target critic.
    self._updates = tf.Variable(0, dtype=tf.int64, trainable=False)
    self.train(next(self.dataset), init_only=True)
    self._update_slow_target(1.0)

  def __call__(self, obs, reset, state=None, training=True):
    step = int(self.step.numpy())
    if state is not None and np.any(reset):
      mask = tf.cast(1 - reset, self.float)[:, None]
      latent, action = state
      latent = {k: v * tf.reshape(mask, [len(reset)] + [1]*(len(v.shape)-1)) for k,v in latent.items()}
      action *= mask
      state = latent, action
    if training and self.should_train(step):
      n = self.c.pretrain if self.should_pretrain() else self.c.train_steps
      for _ in range(n): self.train(next(self.dataset))
      if self.should_log(step): self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training: self.step.assign_add(len(reset) * self.c.action_repeat)
    return action, state

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self.rssm.initial(tf.shape(obs['image'])[0])
      action = tf.zeros([tf.shape(obs['image'])[0], self.actdim], self.float)
    else:
      latent, action = state
    embed = self.encoder(preprocess(obs))
    latent, _ = self.rssm.obs_step(latent, action, embed)
    feat = self.rssm.get_feat(latent)
    dist = self.actor(feat)
    action = dist.sample() if training else dist.mode()
    return action, (latent, action)

  @tf.function
  def train(self, data, init_only=False):
    data = preprocess(data)
    with tf.GradientTape() as model_tape:
      embed = self.encoder(data)
      post, prior = self.rssm.observe(embed, data['action'])
      feat = self.rssm.get_feat(post)
      image_dist = self.decoder(feat)
      reward_dist = self.reward(feat)
      discount_dist = self.discount(feat)
      image_loss = -tf.reduce_mean(image_dist.log_prob(data['image']))
      reward_loss = -tf.reduce_mean(reward_dist.log_prob(data['reward']))
      discount_target = self.c.discount * data['discount']
      discount_loss = -tf.reduce_mean(discount_dist.log_prob(discount_target))
      kl_loss, kl_value = self.rssm.kl_loss(
          post, prior, self.c.kl_balance, self.c.kl_free, False)
      model_loss = image_loss + reward_loss + self.c.discount_scale*discount_loss + self.c.kl_scale*kl_loss
    model_norm = self.model_opt(model_tape, model_loss)

    with tf.GradientTape() as actor_tape:
      imag_feat, imag_action = self._imagine(post)
      reward = self.reward(imag_feat).mean()
      discount = self.discount(imag_feat).mean()
      value = self.slow_critic(imag_feat).mean()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], discount[:-1], value[-1],
          self.c.discount_lambda, axis=0)
      weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(discount[:1]), discount[:-2]], 0), 0))
      # Hybrid parameterized-action gradient split:
      # discrete MOVE/TURN/CATCH -> REINFORCE; continuous parameters -> dynamics gradient.
      actor_dist = self.actor(tf.stop_gradient(imag_feat[:-1]))
      baseline = self.critic(imag_feat[:-1]).mean()
      advantage = tf.stop_gradient(returns - baseline)
      discrete_score = actor_dist.discrete_log_prob(imag_action[:-1]) * advantage
      entropy = actor_dist.entropy()
      dynamics_target = returns
      actor_loss = -tf.reduce_mean(weights * (
          discrete_score + self.c.parameter_grad_scale * dynamics_target
          + self.c.actor_ent * entropy))
    actor_norm = self.actor_opt(actor_tape, actor_loss)

    with tf.GradientTape() as critic_tape:
      critic_dist = self.critic(imag_feat[:-1])
      critic_loss = -tf.reduce_mean(weights * critic_dist.log_prob(tf.stop_gradient(returns)))
    critic_norm = self.critic_opt(critic_tape, critic_loss)

    self._updates.assign_add(1)
    if self.c.slow_target and tf.equal(self._updates % self.c.slow_target_update, 0):
      self._update_slow_target(self.c.slow_target_fraction)
    if not init_only:
      for name, value in dict(model_loss=model_loss, image_loss=image_loss,
          reward_loss=reward_loss, discount_loss=discount_loss, kl=kl_value,
          actor_loss=actor_loss, critic_loss=critic_loss,
          model_grad_norm=model_norm, actor_grad_norm=actor_norm,
          critic_grad_norm=critic_norm).items():
        self.metrics[name].update_state(value)

  def _imagine(self, post):
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k,v in post.items()}
    def step(prev, _):
      feat = self.rssm.get_feat(prev)
      action = self.actor(feat).sample()
      state = self.rssm.img_step(prev, action)
      return state, action
    state = start
    states, actions = [], []
    for _ in range(self.c.imag_horizon):
      feat = self.rssm.get_feat(state)
      action = self.actor(feat).sample()
      state = self.rssm.img_step(state, action)
      states.append(state); actions.append(action)
    states = {k: tf.stack([s[k] for s in states], 0) for k in states[0]}
    actions = tf.stack(actions, 0)
    return self.rssm.get_feat(states), actions

  def _update_slow_target(self, fraction):
    # Variables are created lazily; force both critics once before this call.
    if not self.critic.variables or not self.slow_critic.variables:
      dummy = tf.zeros([1, self.c.rssm_deter + self.c.rssm_stoch*self.c.rssm_discrete], self.float)
      self.critic(dummy); self.slow_critic(dummy)
    for src, dst in zip(self.critic.variables, self.slow_critic.variables):
      dst.assign(fraction*src + (1-fraction)*dst)

  def _write_summaries(self):
    step = int(self.step.numpy())
    values = {k: float(v.result()) for k,v in self.metrics.items()}
    for v in self.metrics.values(): v.reset_state()
    with (self.c.logdir/'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step':step, **values})+'\n')
    print(f'[{step}] ' + ' / '.join(f'{k} {v:.3g}' for k,v in values.items()))


def preprocess(obs):
  obs = obs.copy()
  dtype = prec.global_policy().compute_dtype
  if 'image' in obs:
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
  if 'reward' in obs:
    obs['reward'] = tf.cast(obs['reward'], dtype)
  if 'discount' in obs:
    obs['discount'] = tf.cast(obs['discount'], dtype)
  if 'action' in obs:
    obs['action'] = tf.cast(obs['action'], dtype)
  return obs


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k:v.dtype for k,v in episode.items()}
  shapes = {k:(None,)+v.shape[1:] for k,v in episode.items()}
  gen = lambda: tools.load_episodes(directory, config.train_steps, config.batch_length,
                                    False, capacity=config.replay_capacity)
  sig = {k:tf.TensorSpec(shapes[k], types[k]) for k in types}
  ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
  return ds.batch(config.batch_size, drop_remainder=True).prefetch(config.dataset_prefetch)


def summarize_episode(ep, config, datadir, writer, prefix, progress=None, counters=None):
  """Record one completed episode and update the terminal progress display.

  DreamerV2 still trains from replay sequences and uses environment steps as
  its optimization budget.  This callback only makes the interaction loop
  episode-centric for monitoring, matching the UAV baseline output style.
  """
  length = int((len(ep['reward']) - 1) * config.action_repeat)
  ret = float(np.sum(ep['reward']))
  success = bool(float(np.asarray(ep.get('is_success', [0]))[-1]))
  relay_reached = bool(float(np.asarray(ep.get('relay_reached', [0]))[-1]))
  out_of_bounds = bool(float(np.asarray(ep.get('out_of_bounds', [0]))[-1]))
  terminated = bool(float(np.asarray(ep.get('terminated', [0]))[-1]))
  truncated = bool(float(np.asarray(ep.get('truncated', [0]))[-1]))
  total_steps = int(count_steps(datadir, config))

  if counters is not None:
    counters[prefix] = counters.get(prefix, 0) + 1
    episode = counters[prefix]
  else:
    episode = 0

  record = {
      'step': total_steps,
      f'{prefix}/episode': episode,
      f'{prefix}/return': ret,
      f'{prefix}/length': length,
      f'{prefix}/success': float(success),
      f'{prefix}/relay_reached': float(relay_reached),
      f'{prefix}/out_of_bounds': float(out_of_bounds),
      f'{prefix}/terminated': float(terminated),
      f'{prefix}/truncated': float(truncated),
  }
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(record) + '\n')

  # Only training episodes advance the environment-step progress bar.
  if progress is not None and prefix == 'train':
    progress.update(length)
    progress.set_postfix({
        'ep': episode,
        'ep_steps': length,
        'total_steps': total_steps,
        'success': int(success),
        'relay': int(relay_reached),
        'oob': int(out_of_bounds),
        'return': f'{ret:.0f}',
    }, refresh=True)
  else:
    tqdm.write(
        f'{prefix.title()} EP {episode:05d} | steps={length:3d} | '
        f'total={total_steps:7d} | success={int(success)} | '
        f'relay={int(relay_reached)} | oob={int(out_of_bounds)} | return={ret:.1f}')


def make_env(config, writer, prefix, datadir, store, progress=None, counters=None):
  ctor = functools.partial(wrappers.make_base_env, config.task, config.action_repeat,
                           config.time_limit, config.seed)
  env = wrappers.Async(ctor, config.parallel)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(lambda ep: summarize_episode(
      ep, config, datadir, writer, prefix, progress, counters))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env


def main(config):
  np.random.seed(config.seed); tf.random.set_seed(config.seed)
  if config.gpu_growth:
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  prec.set_global_policy('mixed_float16' if config.precision==16 else 'float32')
  devices=tf.config.list_physical_devices('GPU')
  print('Runtime', devices[0].name if devices else 'CPU', '/ DreamerV2 categorical RSSM')
  config.steps=int(config.steps); config.logdir=pathlib.Path(config.logdir); config.logdir.mkdir(parents=True,exist_ok=True)
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(str(config.logdir))
  initial_steps = count_steps(datadir, config)
  counters = {'train': tools.count_episodes(datadir)[0], 'test': 0}
  progress = tqdm(
      total=config.steps, initial=min(initial_steps, config.steps), unit='step',
      dynamic_ncols=True, desc='DreamerV2 UAV')
  train = [make_env(config, writer, 'train', datadir, True, progress, counters)
           for _ in range(config.envs)]
  test = [make_env(config, writer, 'test', datadir, False, None, counters)
          for _ in range(config.envs)]
  actspace = train[0].action_space
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  tqdm.write(f'Prefill: {prefill} environment steps')
  random_agent = lambda o, d, s: ([actspace.sample() for _ in d], None)
  tools.simulate(random_agent, train, prefill / config.action_repeat)
  agent = DreamerV2(config, datadir, actspace, writer)
  state = None
  step = count_steps(datadir, config)
  while step < config.steps:
    tools.simulate(functools.partial(agent, training=False), test, episodes=1)
    state = tools.simulate(
        agent, train, config.eval_every / config.action_repeat, state=state)
    step = count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')
  progress.close()
  for env in train + test:
    env.close()


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  for key,value in define_config().items():
    parser.add_argument(f'--{key}',type=tools.args_type(value),default=value)
  main(parser.parse_args())
