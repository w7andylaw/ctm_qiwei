"""Replay filtering regressions. Run: python test_replay_sampling.py"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tools


class ReplaySamplingTests(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.directory = Path(self.tmp.name)

  def episode(self, name, length, marker=1):
    np.savez_compressed(self.directory / f'{name}-{length}.npz',
        vector=np.full((length, 13), marker, np.float32),
        reward=np.arange(length, dtype=np.float32))

  def test_short_episode_is_excluded_without_repeated_messages(self):
    self.episode('short', 9, 9)
    self.episode('long', 101, 101)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      sampler = tools.load_episodes(self.directory, 5, length=10)
      for _ in range(100):
        sample = next(sampler)
        self.assertEqual(sample['vector'].shape, (10, 13))
        self.assertTrue(np.all(sample['vector'] == 101))
        np.testing.assert_array_equal(np.diff(sample['reward']), np.ones(9))
    self.assertEqual(output.getvalue().count('Replay filter:'), 1)
    self.assertNotIn('Skipped short episode', output.getvalue())

  def test_all_short_fails_immediately(self):
    self.episode('short', 9)
    with self.assertRaisesRegex(RuntimeError, 'required=10, longest=9, loaded=1'):
      next(tools.load_episodes(self.directory, 1, length=10))

  def test_empty_directory_fails(self):
    with self.assertRaisesRegex(RuntimeError, 'No episodes found'):
      next(tools.load_episodes(self.directory, 1, length=10))

  def test_exact_length_and_full_episode_modes(self):
    self.episode('exact', 10)
    for balance in (False, True):
      sample = next(tools.load_episodes(self.directory, 1, 10, balance))
      np.testing.assert_array_equal(sample['reward'], np.arange(10))
    self.assertEqual(len(next(tools.load_episodes(self.directory, 1))['reward']), 10)

  def test_rescan_occurs_after_yielded_sequence_count(self):
    self.episode('a', 10)
    original_glob = Path.glob
    calls = []
    def counted_glob(path, pattern):
      calls.append(pattern)
      return original_glob(path, pattern)
    with mock.patch.object(Path, 'glob', counted_glob):
      sampler = tools.load_episodes(self.directory, 4, 10)
      for _ in range(4):
        next(sampler)
      self.assertEqual(len(calls), 1)
      self.episode('b', 10, 2)
      next(sampler)
      self.assertEqual(len(calls), 2)
      self.assertTrue(any(np.all(next(sampler)['vector'] == 2) for _ in range(30)))

  def test_capacity_evicts_old_cached_episodes(self):
    self.episode('a', 10, 1)
    sampler = tools.load_episodes(self.directory, 1, 10, capacity=9)
    self.assertTrue(np.all(next(sampler)['vector'] == 1))
    self.episode('b', 10, 2)
    self.assertTrue(np.all(next(sampler)['vector'] == 2))

  def test_summary_changes_only_when_excluded_set_changes(self):
    self.episode('long', 10)
    self.episode('short', 9)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      sampler = tools.load_episodes(self.directory, 1, 10)
      next(sampler)
      self.episode('another_long', 20)
      next(sampler)
      self.episode('another_short', 8)
      next(sampler)
      next(sampler)
    self.assertEqual(output.getvalue().count('Replay filter:'), 2)

  def test_zero_rescan_fails_instead_of_spinning(self):
    with self.assertRaisesRegex(ValueError, 'rescan must be at least 1'):
      next(tools.load_episodes(self.directory, 0, 10))


if __name__ == '__main__':
  unittest.main(verbosity=2)
