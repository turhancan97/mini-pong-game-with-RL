# Deep Q-Learning for Pong

## Abstract
This project implements a modular Deep Q-Learning (DQL) agent that learns to play Pong directly from pixels. The repository combines a lightweight pygame environment, convolutional neural networks built with TensorFlow/Keras, and Hydra-based experiment management. Training leverages experience replay, a soft-updated target network, and configurable logging utilities to facilitate reproducible research workflows.

## Table of Contents
- [Deep Q-Learning for Pong](#deep-q-learning-for-pong)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Repository Layout](#repository-layout)
  - [Installation](#installation)
  - [Running Experiments](#running-experiments)
    - [Training](#training)
    - [Inference](#inference)
  - [Configuration](#configuration)
  - [Outputs and Logging](#outputs-and-logging)
  - [Implementation Notes](#implementation-notes)
  - [Reproducibility Checklist](#reproducibility-checklist)
  - [References](#references)

## Introduction
The project investigates off-policy reinforcement learning for a classic control problem. An agent observes stacked grayscale frames (40x40) and selects discrete paddle actions. Training minimises the temporal-difference error between online and target Q-networks, while exploration is controlled by a time-dependent epsilon schedule. The modular design allows the environment, agent, and training loop to evolve independently.

## Repository Layout
- `pong_rl/agent.py`: Deep Q-Network agent with replay buffer, epsilon-greedy policy, and Polyak-averaged target updates.
- `pong_rl/environment.py`: Headless-friendly Pong simulator built on pygame.
- `pong_rl/processing.py`: Frame preprocessing (cropping, grayscale, resizing, intensity scaling).
- `pong_rl/replay_buffer.py`: Fixed-capacity experience store for sampled mini-batches.
- `pong_rl/train_loop.py`: Shared utilities for stacking frames and recording training metrics.
- `pong_rl/utils.py`: Convenience helpers (e.g., global random seeding).
- `conf/config.yaml`: Default Hydra configuration for paths, agent hyperparameters, and training schedules.
- `train.py`: Training entry point with resumable checkpoints, score serialization, and optional matplotlib plotting.
- `inference.py`: Greedy policy rollout with rendering enabled for qualitative inspection.
- `environment.yml`: Conda specification capturing the full software stack.

## Installation
1. Install [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/).
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate pong-rl
   ```
3. (Optional, Apple Silicon) For accelerated training, install `tensorflow-metal` inside the activated environment:
   ```bash
   pip install tensorflow-metal
   ```

## Running Experiments
### Training
```bash
python train.py
```
Key artifacts after training:
- SavedModel directory in `models/latest_model`
- Replay-aware checkpoint in `checkpoints/latest`
- Score history (`scores.pkl`) and optional plot (`scores.png`) in `plots`

Resume from the latest checkpoint automatically (default `training.resume=true`). To start afresh, either delete the checkpoint directory or launch with `python train.py training.resume=false`.

### Inference
After at least one successful training run:
```bash
python inference.py
```
This launches pygame with the agent acting greedily while streaming console metrics every 25 steps.

## Configuration
Hydra enables command-line overrides without editing config files. Examples:
- Shorten training: `python train.py training.total_steps=50000`
- Disable plotting: `python train.py logging.plot_scores=false`
- Adjust target network smoothing: `python train.py agent.target_update_tau=0.05`
- Change seed or determinism: `python train.py seed=123 reproducibility.enable_tf_determinism=false`

All default parameters are documented in `conf/config.yaml`. Additional Hydra features (multi-run sweeps, output directories, etc.) can be used if you re-enable Hydra logging/output in the config.

## Outputs and Logging
- **Checkpoints**: `checkpoints/latest/model/` (SavedModel) and `checkpoints/latest/agent_state.pkl` (replay buffer, epsilon, counters).
- **Final Model**: `models/latest_model`
- **Training History**: `plots/scores.pkl` (steps, smoothed scores, losses) and, when enabled, `plots/scores.png`.
- **Console Logs**: Progress every `training.log_interval` steps with reward, smoothed score, and loss.

## Implementation Notes
- Convolutional architecture: three stacked Conv2D layers followed by a 512-unit dense head (see `pong_rl/agent.py:55`).
- Replay warm-up prevents premature training until the buffer accumulates `agent.min_replay_size` transitions.
- Target network weights are soft-updated every `agent.target_update_interval` training steps using Polyak averaging.
- Frame preprocessing mirrors classic Atari DQN pipelines (grayscale, downsampling, and intensity normalization).
- Headless training sets `SDL_VIDEODRIVER=dummy`, enabling server-side execution without opening windows.
- Global seeding (configurable via `seed` and `reproducibility.enable_tf_determinism`) keeps experiments reproducible.

## Reproducibility Checklist
- [x] Version-controlled configuration (`conf/config.yaml`)
- [x] Conda environment specification (`environment.yml`)
- [x] Deterministic preprocessing pipeline
- [x] Checkpointing with replay buffer serialization
- [x] Fixed random seeds (configurable via `seed` in `conf/config.yaml`)

## References
1. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 2015.
2. Yadav et al., "Hydra: A framework for elegantly configuring complex applications," 2020.
3. TensorFlow Core Team, "TensorFlow 2.0: Large-scale machine learning on heterogeneous systems," 2019.
