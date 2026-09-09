import atexit
import functools
import os
import pickle
import subprocess
import sys
import threading
import traceback

try:
  import gymnasium as gym
except ImportError:  # Backward compatibility with the original environment.
  import gym
import numpy as np
from PIL import Image


def make_base_env(task, action_repeat, time_limit, seed=None):
  """Build an environment without importing TensorFlow in the worker."""
  suite, task = task.split('_', 1)
  if suite == 'dmc':
    env = DeepMindControl(task)
    env = ActionRepeat(env, action_repeat)
    env = NormalizeActions(env)
  elif suite == 'atari':
    env = Atari(
        task, action_repeat, (64, 64), grayscale=False,
        life_done=True, sticky_actions=True)
    env = OneHotAction(env)
  elif suite == 'uav':
    from envs import DreamerV2UAVEnv
    env = DreamerV2UAVEnv(task, max_episode_steps=time_limit, seed=seed)
    env = ActionRepeat(env, action_repeat)
  else:
    raise NotImplementedError(suite)
  return TimeLimit(env, time_limit / action_repeat)


class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(
        spec.minimum.astype(np.float32), spec.maximum.astype(np.float32),
        dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True):
    import gym
    version = 0 if sticky_actions else 4
    name = ''.join(word.title() for word in name.split('_'))
    with self.LOCK:
      self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
    self._action_repeat = action_repeat
    self._size = size
    self._grayscale = grayscale
    self._noops = noops
    self._life_done = life_done
    self._lives = None
    shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    self._random = np.random.RandomState(seed=None)

  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()
    noops = self._random.randint(1, self._noops + 1)
    for _ in range(noops):
      done = self._env.step(0)[2]
      if done:
        with self.LOCK:
          self._env.reset()
    self._lives = self._env.ale.lives()
    if self._grayscale:
      self._env.ale.getScreenGrayscale(self._buffers[0])
    else:
      self._env.ale.getScreenRGB2(self._buffers[0])
    self._buffers[1].fill(0)
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward
      if self._life_done:
        lives = self._env.ale.lives()
        done = done or lives < self._lives
        self._lives = lives
      if done:
        break
      elif step >= self._action_repeat - 2:
        index = step - (self._action_repeat - 2)
        if self._grayscale:
          self._env.ale.getScreenGrayscale(self._buffers[index])
        else:
          self._env.ale.getScreenRGB2(self._buffers[index])
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    if self._action_repeat > 1:
      np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
    image = np.array(Image.fromarray(self._buffers[0]).resize(
        self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}


class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, blocking=True):
    result = self._env.step(action, blocking=blocking)
    if not blocking:
      return lambda: self._process_step(result(), action)
    return self._process_step(result, action)

  def _process_step(self, result, action):
    obs, reward, done, info = result
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self, blocking=True):
    result = self._env.reset(blocking=blocking)
    if not blocking:
      return lambda: self._process_reset(result())
    return self._process_reset(result)

  def _process_reset(self, obs):
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class ObsDict:

  def __init__(self, env, key='obs'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = dict(self._env.observation_space.spaces)
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(
        -np.inf, np.inf, shape=(), dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action, blocking=True):
    result = self._env.step(action, blocking=blocking)
    if not blocking:
      return lambda: self._process_step(result())
    return self._process_step(result)

  def _process_step(self, result):
    obs, reward, done, info = result
    obs['reward'] = np.asarray(reward, np.float32)
    return obs, reward, done, info

  def reset(self, blocking=True):
    result = self._env.reset(blocking=blocking)
    if not blocking:
      return lambda: self._process_reset(result())
    return self._process_reset(result)

  def _process_reset(self, obs):
    obs['reward'] = np.asarray(0.0, np.float32)
    return obs


class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      env = os.environ.copy()
      paths = env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
      paths = [path for path in paths if 'site-packages/nvidia' not in path]
      env['LD_LIBRARY_PATH'] = os.pathsep.join(paths)
      self._process = subprocess.Popen(
          [sys.executable, '-u', __file__, '--worker'],
          stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
      import cloudpickle
      cloudpickle.dump(ctor, self._process.stdin)
      self._process.stdin.flush()
    else:
      raise NotImplementedError(strategy)
    if strategy == 'thread':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=_async_worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    elif strategy == 'process':
      atexit.register(self.close)
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      promise = functools.partial(getattr(self._env, name), *args, **kwargs)
      return promise() if blocking else promise
    payload = name, args, kwargs
    self._send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._send((self._CLOSE, None))
      if self._strategy == 'process':
        self._process.stdin.close()
        self._process.stdout.close()
      else:
        self._conn.close()
    except (IOError, OSError, ValueError):
      # The connection was already closed.
      pass
    if self._strategy == 'process':
      self._process.wait()
    else:
      self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      if self._strategy == 'process':
        message, payload = pickle.load(self._process.stdout)
      else:
        message, payload = self._conn.recv()
    except (ConnectionResetError, EOFError):
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _send(self, message):
    if self._strategy == 'process':
      pickle.dump(message, self._process.stdin)
      self._process.stdin.flush()
    else:
      self._conn.send(message)



def _async_worker(ctor, conn):
  """Spawn-safe worker kept outside Async to avoid pickling the parent."""
  try:
    env = ctor()
    while True:
      try:
        if not conn.poll(0.1):
          continue
        message, payload = conn.recv()
      except (EOFError, KeyboardInterrupt):
        break
      if message == Async._ACCESS:
        conn.send((Async._RESULT, getattr(env, payload)))
        continue
      if message == Async._CALL:
        name, args, kwargs = payload
        result = getattr(env, name)(*args, **kwargs)
        conn.send((Async._RESULT, result))
        continue
      if message == Async._CLOSE:
        assert payload is None
        break
      raise KeyError(f'Received message of unknown type {message}')
  except Exception:
    stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
    print(f'Error in environment process: {stacktrace}')
    conn.send((Async._EXCEPTION, stacktrace))
  conn.close()


def _subprocess_worker():
  """Environment worker launched as a clean interpreter."""
  import cloudpickle
  stream_in = sys.stdin.buffer
  stream_out = sys.stdout.buffer
  ctor = cloudpickle.load(stream_in)
  env = ctor()
  while True:
    try:
      message, payload = pickle.load(stream_in)
    except EOFError:
      break
    try:
      if message == Async._ACCESS:
        result = getattr(env, payload)
      elif message == Async._CALL:
        name, args, kwargs = payload
        result = getattr(env, name)(*args, **kwargs)
      elif message == Async._CLOSE:
        break
      else:
        raise KeyError(f'Received message of unknown type {message}')
      pickle.dump((Async._RESULT, result), stream_out)
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      pickle.dump((Async._EXCEPTION, stacktrace), stream_out)
    stream_out.flush()


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
  _subprocess_worker()
