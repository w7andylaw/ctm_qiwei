# Dreamer walker 300k 对照实验

三组实验使用相同的 `dmc_walker_walk`、seed 1、FP32 和 300,000 步：

| 组别 | batch size | replay capacity | 目的 |
|---|---:|---:|---|
| `control` | 32 | 100,000 | 当前本机配置的独立复现 |
| `replay_all` | 32 | 0（全部历史） | 只测 replay 上限的影响 |
| `replay_all_batch50` | 50 | 0（全部历史） | 在全部 replay 上只测 batch 的影响 |

在 WSL Ubuntu 终端运行全部实验：

```bash
cd /mnt/e/CTM/Dreamer-master
bash ablations/walker_300k/run_comparison.sh
```

在 Windows PowerShell 终端运行：

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/CTM/Dreamer-master && bash ablations/walker_300k/run_comparison.sh"
```

实验使用固定目录；中断后重复同一命令会从各自 checkpoint 继续。也可以只运行指定组：

```bash
bash ablations/walker_300k/run_comparison.sh replay_all
```

结果写入：

```text
logdir/ablations/walker_300k_seed1/
├── control/
├── replay_all/
├── replay_all_batch50/
├── comparison.csv
└── comparison.png
```

若 batch 50 显存不足，该组会标记为失败，前两组结果仍会保留并汇总。不要复用现有的 `logdir/dmc_walker_walk/dreamer/1`，以免污染原训练结果。

## 加速模式

最安全的加速方式是把高频读写的 `logdir` 放到 WSL 的 ext4 文件系统，并适当增加数据预取；这不改变环境数量：

```bash
mkdir -p /home/madao/ctm_runs
CTM_ABLATION_LOGDIR=/home/madao/ctm_runs/walker_300k_seed1 \
CTM_ABLATION_PREFETCH=6 \
bash ablations/walker_300k/run_comparison.sh
```

速度优先时可以并行运行 4 个环境，但这会改变相对原始单环境基准的运行配置。三组对照之间仍然公平，因为它们使用相同的环境数：

```bash
CTM_ABLATION_LOGDIR=/home/madao/ctm_runs/walker_300k_seed1_env4 \
CTM_ABLATION_PREFETCH=6 \
CTM_ABLATION_ENVS=4 \
bash ablations/walker_300k/run_comparison.sh
```

可选环境变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `CTM_ABLATION_STEPS` | 300000 | 每组训练步数 |
| `CTM_ABLATION_SEED` | 1 | 三组共用的随机种子 |
| `CTM_ABLATION_PREFETCH` | 2 | TensorFlow 数据预取批数 |
| `CTM_ABLATION_ENVS` | 1 | 并行环境数 |
| `CTM_ABLATION_LOGDIR` | 项目内 `logdir` | 实验输出根目录 |
