<div align="center">

# Latent Dynamics Geometries

[![Paper](https://img.shields.io/badge/Paper-OpenReview-B31B1B?logo=openreview&logoColor=white)](https://openreview.net/forum?id=XQLa5PVQ0D)
[![arXiv](https://img.shields.io/badge/arXiv-2606.02280-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.02280)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries For Zero-Shot Policy Adaptation**

<img src="assets/poster_ldg_icml2026.png" alt="Latent Dynamics Geometries overview" width="900"/>

</div>

This repository provides a modular reinforcement learning framework for **dynamics adaptation** in continuous-control. It ships our method — [Latent Dynamics Geometries (LDG)](https://arxiv.org/abs/2606.02280) — alongside baseline implementations of [RMA](https://arxiv.org/abs/2107.04034) and [SO-CMA](https://arxiv.org/abs/1702.02453), all sharing a unified training, adaptation, and evaluation pipeline. Agents, environments, and wrappers are composed entirely through YAML configuration for reproducibility and extensibility.

## Features

- **Multiple adaptation methods in one codebase.** Includes LDG, RMA, and SO-CMA baselines, all sharing the same training and evaluation infrastructure.
- **Unified, modular design.** Methods and environments are organized into a single framework with a consistent pipeline. Swap agents or environment wrappers by editing `configs/user_config.yaml` — no code changes required.
- **Full evaluation suite.** Supports in-distribution, out-of-distribution (OOD), structural-failure, and time-varying dynamics evaluation across MuJoCo locomotion tasks (Hopper, Walker2d, HalfCheetah, Ant).

## Results

Below is our official LDG results combining in-distribution and zero-shot generalization rewards from the paper. Rewards are mean cumulative return; in-distribution results additionally report ± one standard deviation over 5 dynamics sets. OOD columns sweep an physical property beyond the training range (mass for Hopper/HalfCheetah, damping for Walker2d/Ant).

| Environment | In-Distribution | OOD (low scale) | OOD (high scale) | Var. Env | Struct. Fail. |
|-------------|----------------:|----------------:|-----------------:|---------:|--------------:|
| Hopper | 3239.2 ± 279.3 | 3154 (0.5× mass) | 1349 (2.0× mass) | 3294 | 596 |
| Walker2d | 4883.7 ± 381.6 | 5342 (0.3× damping) | 5349 (2.2× damping) | 5293 | 952 |
| HalfCheetah | 5799.1 ± 1053.4 | 5999 (0.5× mass) | 5284 (2.0× mass) | 7849 | 2071 |
| Ant | 5183.9 ± 296.9 | 4797 (0.3× damping) | 4789 (2.2× damping) | 3462 | 1003 |

**Scenario definitions:**
- **In-distribution** — asymptotic performance on dynamics parameters within the training range
- **OOD** — parameter tested at out-of-range scale factors (e.g., training range [0.5, 2.0), test at 0.3 and 2.2)
- **Var. Env** — dynamics parameters shift mildly every 200 steps within an episode
- **Struct. Fail.** — a joint is disabled mid-episode via zero-command enforcement ([`configs/environments/adapt/adapt_env.yaml`](configs/environments/adapt/adapt_env.yaml))

### Walker2d hopping gait

The official Walker2d row above — and the paper's RMA Walker2d baselines — were trained **without** [`LimitAngleTorqueWrapper`](ldr/environments/wrappers/local_wrappers/limit_angle_torque_wrapper.py). That setup produces very high return, but the policy typically learns a **hopping** gait rather than a walk. Turning the wrapper on clips ankle torque to 20% of the default range and suppresses hopping. We report the paper numbers from the no-wrapper setting because **RMA fails to train** once ankle torque is limited. LDG still trains in that stricter environment, with results provided below:

| Environment | In-Distribution | OOD (0.3× damping) | OOD (2.2× damping) | Var. Env | Struct. Fail. |
|-------------|----------------:|-------------------:|-------------------:|---------:|--------------:|
| Walker2d (ankles limited) | 4143.6 ± 318.3 | 4527 | 4521 | 3695 | 442 |

To match the paper / hopping setup, comment out `LimitAngleTorqueWrapper` in [`configs/environments/walker2d.yaml`](configs/environments/walker2d.yaml). Leave it enabled to enforce walking gait.

## Repository Structure

```
Latent-Dynamics-Geometries/
├── configs/
│   ├── user_config.yaml     # Point here to choose agent + environment for training
│   ├── agents/              # Per-task agent configs (ldr, rma, sac)
│   ├── environments/        # MuJoCo env + dynamics wrapper configs
│   └── adapt/               # Adaptation module configs
├── ldr/                     # Core package
│   ├── agents/              # LDR, RMA, SAC implementations
│   ├── adaptation/          # RMA phase-2 adaptation
│   ├── core/                # Trainer and evaluators
│   ├── environments/        # Vec envs and physics wrappers
│   └── visualizer/          # Reward plotting and embedding visualization
└── scripts/
    ├── train/               # Training entry point
    ├── adapt/               # Adaptation entry point
    ├── exp/                 # Evaluation scripts
    └── visualization/       # Latent and state-prediction visualization
```

## Getting Started

### Installation

Requires [Conda](https://docs.conda.org/) and a CUDA-capable GPU (optional but recommended).

```bash
bash setup.sh
conda activate ldr
```

This creates a conda environment named `ldr` with Python 3.10. By default it installs `torch==2.3.1` built for **CUDA 12.1**. Adjust the PyTorch install line in `setup.sh` if your CUDA version differs.

### 1. Configure and Train

Edit [`configs/user_config.yaml`](configs/user_config.yaml) to select the environment and agent:

```yaml
environment: ./environments/hopper.yaml
agent: ./agents/hopper/ldr.yaml
```

Then train:

```bash
cd scripts/train
python train.py
```

Training logs are written under `logs/<agent name>` (e.g., `logs/ldr`). Episode rewards are recorded by [`RewardMonitor`](ldr/agents/utils/reward_monitor.py) into `monitor.csv`. You can also use TensorBoard (logs are saved alongside checkpoints).

To plot reward curves:

```bash
MONITOR_DIR=logs/<experiment_dir>/monitor bash scripts/visualization/plot_rewards.sh
```

### 2. Adapt (RMA)

For RMA agents, run phase-2 adaptation on a trained experiment directory. The script loads the agent and environment YAMLs from that run's `backup/` folder and selects the highest-reward checkpoint under `ckpt/`:

```bash
EXP_DIR=/path/to/experiment_run_dir bash scripts/adapt/rma_adapt.sh
```

The adaptation config is at [`configs/adapt/rma_adapt.yaml`](configs/adapt/rma_adapt.yaml). Agent and environment come from the experiment backup.

### 3. Evaluate

Evaluation scripts expect a **parent directory** containing multiple experiment runs. Each run directory should follow this layout:

```
<base_exp_dir>/
└── 20260121_151039_LDRAgent_Hopper/
    ├── backup/
    │   ├── ldr.yaml          # backed-up agent config
    │   └── hopper.yaml       # backed-up environment config
    └── ckpt/
        └── LDRAgent_reward_3500.00.pth
```

Run evaluation from the `scripts/` directory:

```bash
# LDR agent
BASE_EXP_DIR=/path/to/experiment_parent_dir bash eval.sh

# RMA with SO-CMA adaptation
BASE_EXP_DIR=/path/to/experiment_parent_dir bash eval_cma.sh

# RMA with phase-2 adaptation module
BASE_EXP_DIR=/path/to/experiment_parent_dir bash eval_rma.sh
```

Set `AGENT=<name>` to override the script default (`ldr` for `eval.sh`, `rma` otherwise). Set `MODE=varenv` for time-varying dynamics evaluation:

```bash
MODE=varenv BASE_EXP_DIR=/path/to/experiment_parent_dir bash eval.sh
```

Agent and environment YAMLs are loaded from each run's `backup/` directory. RMA Phase 2 uses [`configs/adapt/rma_adapt.yaml`](configs/adapt/rma_adapt.yaml); SO-CMA uses [`configs/adapt/cma_adapt.yaml`](configs/adapt/cma_adapt.yaml).

**Evaluation modes (`MODE=normal`):**
- **In-distribution** — dynamics randomized every episode
- **OOD** — sweep mass or damping outside training range
- **Structural failure** — uses [`adapt_env.yaml`](configs/environments/adapt/adapt_env.yaml) (crippled joints, reward lag)

**Evaluation modes (`MODE=varenv`):**
- **Time-varying dynamics** — parameters shift mid-episode

### 4. Visualize

Two visualization scripts are provided for analyzing a trained LDG checkpoint. Ensure [`configs/user_config.yaml`](configs/user_config.yaml) matches the agent and environment used during training.

**Latent embedding structure** — reproduces the style of [Figure 2](https://arxiv.org/html/2606.02280) in the paper. Sweeps a physical parameter (e.g., mass or damping) across in- and out-of-distribution values, collects trajectory latents, and projects them with t-SNE to reveal whether the encoder forms a smooth, monotonic manifold:

```bash
MODEL_PATH=/path/to/agent.pth bash scripts/visualization/vis_embed.sh
```

**Decoder state prediction** — visualizes the conditional VAE decoder's next-state predictions against ground truth over a rollout, showing how well the learned dynamics model reconstructs state transitions under the inferred latent context:

```bash
MODEL_PATH=/path/to/agent.pth bash scripts/visualization/vis_state.sh
```

## Acknowledgements

This project builds upon:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — `DummyVecEnv` implementation and SAC reference

## Citation

If you find this work useful, please cite:

```bibtex
@article{latent-dynamics-geometries,
  title={Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries For Zero-Shot Policy Adaptation},
  author={Xu, Zhiming and Zhou, Weitao and Pan, Xianghui and Deng, Nanshan and Liu, Chengju and Chen, Qijun and Yao, Chenpeng},
  journal={arXiv preprint arXiv:2606.02280},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
