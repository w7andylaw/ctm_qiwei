"""Regression checks for state/action alignment and the navigation contract."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import unittest
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
import models
import tools
import dreamer
from envs import DreamerV2UAVEnv


class NavigationFixes(unittest.TestCase):
  def test_vector_retains_speed_and_steps_when_image_is_identical(self):
    env = DreamerV2UAVEnv('relay')
    obs, info = env.env.reset(seed=3)
    before = env._convert_obs(obs, info)
    env.env.state.speed = 17.0
    env.env.elapsed_steps = 30
    after = env._convert_obs(env.env._get_obs(), info)
    np.testing.assert_array_equal(before['image'], after['image'])
    self.assertEqual(after['vector'].shape, (13,))
    self.assertNotEqual(before['vector'][2], after['vector'][2])
    self.assertNotEqual(before['vector'][9], after['vector'][9])
    self.assertTrue(env.observation_space['vector'].contains(after['vector']))

  def test_action_masks_and_gradients(self):
    action = tf.Variable([[1.,0.,0.,.2,.7,.9], [0.,0.,1.,.4,.5,.6]])
    with tf.GradientTape() as tape:
      actual = models.canonical_action(action)
      loss = tf.reduce_sum(actual[...,3:])
    np.testing.assert_allclose(actual.numpy(), [[1,0,0,.2,0,0],[0,0,1,0,0,0]])
    np.testing.assert_allclose(tape.gradient(loss, action).numpy()[...,3:], [[1,0,0],[0,0,0]])
    np.testing.assert_array_equal(models.canonical_action(tf.zeros([1,6])), np.zeros([1,6]))
    env = DreamerV2UAVEnv('relay')
    self.assertEqual(env.decode_action(action.numpy()[0])[0], env.decode_action(actual.numpy()[0])[0])
    np.testing.assert_allclose(env.decode_action(action.numpy()[0])[1], env.decode_action(actual.numpy()[0])[1])

  def test_rssm_ignores_unexecuted_parameters(self):
    rssm = models.RSSM(stoch=3, deter=8, hidden=8, discrete=4)
    state = rssm.initial(1)
    first = rssm.img_step(state, tf.constant([[1.,0.,0.,.2,.7,.9]]))
    second = rssm.img_step(state, tf.constant([[1.,0.,0.,.2,-.3,-.4]]))
    np.testing.assert_allclose(first['logits'], second['logits'])
    np.testing.assert_allclose(first['deter'], second['deter'])

  def test_imagination_pairs_actions_with_source_states(self):
    class Dynamics:
      def get_feat(self, state): return state['deter']
      def img_step(self, state, action): return {'deter': state['deter'] + 1}
    class Distribution:
      def __init__(self, feat): self.feat = feat
      def sample(self): return self.feat * 10
    class Agent:
      c = tools.AttrDict(imag_horizon=3)
      rssm = Dynamics()
      actor = staticmethod(Distribution)
    features, actions = dreamer.DreamerV2._imagine(Agent(), {'deter': tf.constant([[[2.]]])})
    np.testing.assert_allclose(features[:,0,0], [2,3,4,5])
    np.testing.assert_allclose(actions[:,0,0], [20,30,40])
    # Arrival rewards [1,2,3] and bootstrap V(s_3)=4: 3+4, 2+7, 1+9.
    returns = tools.lambda_return(tf.constant([[1.],[2.],[3.]]),
        tf.zeros([3,1]), tf.ones([3,1]), tf.constant([4.]), 1.0, axis=0)
    np.testing.assert_allclose(returns[:,0], [10,9,7])

  def test_sparse_reward_has_no_early_failure_bonus(self):
    for task in ('direct', 'relay'):
      env = DreamerV2UAVEnv(task)
      options = dict(start=[1999,1000], heading=0., speed=40.)
      options.update(dict(goal=[100,100]) if task == 'direct' else dict(relay_goal=[100,100],final_goal=[300,300]))
      env.env.reset(seed=1, options=options)
      a = np.zeros(2*env.num_actions, np.float32); a[0] = 1
      _, reward, done, info = env.step(a)
      self.assertTrue(done and info['out_of_bounds'])
      self.assertEqual(reward, 0.)
      options['start'] = [1000,1000]; options['speed'] = 0.
      env.env.reset(seed=1, options=options)
      total = 0.
      for _ in range(100):
        _, reward, done, info = env.step(a); total += reward
        if done: break
      self.assertTrue(info['truncated'])
      self.assertEqual(total, 0.)
    env = DreamerV2UAVEnv('direct')
    env.env.reset(options=dict(start=[990,1000],goal=[1000,1000],speed=0.,heading=0.))
    _, reward, done, info = env.step(np.asarray([1,0,0,0], np.float32))
    self.assertTrue(done and info['is_success']); self.assertEqual(reward, 1.)
    env = DreamerV2UAVEnv('relay')
    env.env.reset(options=dict(start=[500,500],relay_goal=[500,500],final_goal=[1500,1500],speed=0.,heading=0.))
    obs, reward, done, info = env.step(np.asarray([0,0,1,.2,.5,.8], np.float32))
    self.assertEqual(reward, 0.); self.assertEqual(obs['vector'][10], 1.)
    env.env.state.x = 1500; env.env.state.y = 1500
    _, reward, done, info = env.step(np.asarray([1,0,0,0,0,0], np.float32))
    self.assertTrue(done and info['is_success']); self.assertEqual(reward, 1.)

  def test_training_and_policy_without_images(self):
    # Real replay, model/actor/critic updates, and policy inference on vector-only data.
    for task in ('direct', 'relay'):
      with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = DreamerV2UAVEnv(task, max_episode_steps=10)
        obs = env.reset()
        records = [dict(vector=obs['vector'],action=np.zeros(2*env.num_actions,np.float32),reward=0.,discount=1.)]
        for _ in range(10):
          action = np.zeros(2*env.num_actions,np.float32); action[0] = 1
          obs, reward, done, info = env.step(action)
          records.append(dict(vector=obs['vector'],action=action,reward=reward,discount=info['discount']))
          if done: break
        episode = {k:np.asarray([r[k] for r in records],np.float32) for k in records[0]}
        tools.save_episodes(root/'episodes', [episode])
        c = dreamer.define_config(); c.logdir = root; c.task = 'uav_'+task
        c.batch_size=2; c.batch_length=4; c.imag_horizon=3; c.num_units=16
        c.rssm_stoch=3; c.rssm_discrete=4; c.rssm_deter=16; c.rssm_hidden=16
        agent = dreamer.DreamerV2(c, root/'episodes', env.action_space, None)
        agent.train(next(agent.dataset))
        for metric in agent.metrics.values(): self.assertTrue(np.isfinite(metric.result().numpy()))
        action, state = agent.policy({'vector':tf.constant(obs['vector'][None])},None,False)
        self.assertEqual(action.shape, (1,2*env.num_actions))
        self.assertTrue(np.isfinite(action.numpy()).all())


if __name__ == '__main__': unittest.main(verbosity=2)
