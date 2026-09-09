"""Summarize Dreamer walker ablations and plot them against bundled scores."""

import argparse
import csv
import json
import pathlib
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS = {
    'control': 'A: batch32 + replay100k',
    'replay_all': 'B: batch32 + all replay',
    'replay_all_batch50': 'C: batch50 + all replay',
}


def load_metrics(path):
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text().splitlines():
    if line.strip():
      rows.append(json.loads(line))
  return rows


def mean(values):
  return statistics.fmean(values) if values else float('nan')


def summarize(name, rows):
  train = [row['train/return'] for row in rows if 'train/return' in row]
  test = [row for row in rows if 'test/return' in row]
  return {
      'variant': name,
      'max_step': int(max((row.get('step', 0) for row in rows), default=0)),
      'train_last100_mean': mean(train[-100:]),
      'test_latest': test[-1]['test/return'] if test else float('nan'),
      'test_last5_mean': mean([row['test/return'] for row in test[-5:]]),
      'test_best': max((row['test/return'] for row in test), default=float('nan')),
      'test_best_step': int(max(
          test, key=lambda row: row['test/return'], default={'step': 0})['step']),
  }


def load_reference(path, max_step):
  runs = [
      run for run in json.loads(path.read_text())
      if run['task'] == 'dmc_walker_walk']
  by_step = {}
  for run in runs:
    for step, score in zip(run['xs'], run['ys']):
      if step <= max_step:
        by_step.setdefault(step, []).append(score)
  xs = sorted(step for step, values in by_step.items() if len(values) == len(runs))
  means = [mean(by_step[step]) for step in xs]
  lows = [min(by_step[step]) for step in xs]
  highs = [max(by_step[step]) for step in xs]
  return xs, means, lows, highs


def fmt(value):
  return '-' if value != value else f'{value:.1f}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logroot', type=pathlib.Path, required=True)
  parser.add_argument('--scores', type=pathlib.Path, required=True)
  args = parser.parse_args()

  args.logroot.mkdir(parents=True, exist_ok=True)
  loaded = {}
  summaries = []
  for name in LABELS:
    rows = load_metrics(args.logroot / name / 'metrics.jsonl')
    if rows:
      loaded[name] = rows
      summaries.append(summarize(name, rows))

  csv_path = args.logroot / 'comparison.csv'
  fields = [
      'variant', 'max_step', 'train_last100_mean', 'test_latest',
      'test_last5_mean', 'test_best', 'test_best_step']
  with csv_path.open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summaries)

  print(
      f"{'Variant':<25} {'Step':>8} {'Train100':>10} "
      f"{'TestNow':>9} {'Test5':>9} {'Best':>9}")
  for item in summaries:
    print(
        f"{LABELS[item['variant']]:<25} {item['max_step']:>8} "
        f"{fmt(item['train_last100_mean']):>10} "
        f"{fmt(item['test_latest']):>9} "
        f"{fmt(item['test_last5_mean']):>9} "
        f"{fmt(item['test_best']):>9}")

  if not loaded:
    print('No completed metrics found yet.')
    return

  max_step = max(item['max_step'] for item in summaries)
  fig, ax = plt.subplots(figsize=(9, 5.5))
  for name, rows in loaded.items():
    test = [row for row in rows if 'test/return' in row]
    ax.plot(
        [row['step'] for row in test],
        [row['test/return'] for row in test],
        linewidth=1.7, label=LABELS[name])

  xs, means, lows, highs = load_reference(args.scores, max_step)
  if xs:
    ax.fill_between(xs, lows, highs, alpha=0.12, color='black')
    ax.plot(xs, means, '--', color='black', linewidth=1.6,
            label='Bundled reference: 5-seed mean/range')

  ax.set_title('Dreamer dmc_walker_walk ablation')
  ax.set_xlabel('Environment steps')
  ax.set_ylabel('Test return')
  ax.set_xlim(left=0)
  ax.set_ylim(0, 1000)
  ax.grid(alpha=0.25)
  ax.legend(loc='best')
  fig.tight_layout()
  fig.savefig(args.logroot / 'comparison.png', dpi=160)
  plt.close(fig)
  print(f'CSV: {csv_path}')
  print(f'Plot: {args.logroot / "comparison.png"}')


if __name__ == '__main__':
  main()
