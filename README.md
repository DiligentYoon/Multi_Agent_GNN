# Connectivity-Aware Multi Robot Exploration and Navigation in Cluttered Corridors Using Graph Attention Network

This project implements a Multi-Agent Reinforcement Learning (MARL) framework for navigation tasks using Graph Neural Networks (GNN) and Proximal Policy Optimization (PPO). It features a centralized learner with distributed rollout workers using Ray, and supports multiple model architectures including a specialized "CoMapping" policy.

## Project Structure

```
├── main_driver.py          # Main entry point for training
├── test_main_driver.py     # Script for testing and visualization
├── config/                 # Configuration files
│   └── nav_ppo_cfg.yaml    # Main configuration for PPO and environment
├── task/                   # Core source code
│   ├── agent/              # RL Agents (PPO, SAC)
│   ├── env/                # Navigation Environment (NavEnv)
│   ├── model/              # Neural Network Architectures (GNN, Actor-Critic)
│   ├── worker/             # Ray Rollout Workers
│   └── buffer/             # Rollout Buffers
└── results/                # Output directory for logs and checkpoints
```

## Prerequisites

Ensure you have the following installed:

*   Python 3.8+
*   [Ray](https://www.ray.io/)
*   [PyTorch](https://pytorch.org/)
*   [Gymnasium](https://gymnasium.farama.org/)
*   NumPy, Pandas, Matplotlib, ImageIO, PyYAML, TensorBoard

## Configuration

The main configuration file is located at `config/nav_ppo_cfg.yaml`. It controls:

*   **Env**: Number of agents, map properties, sensor settings.
*   **Agent**: PPO hyperparameters (learning rate, clip ratio, etc.).
*   **Train**: Total timesteps, evaluation frequency.
*   **Model**: Architecture type (GNN), layers, and history usage.

## Usage

### Training

To start a training session, run `main_driver.py`. You can specify the model version and optional checkpoint to resume from.

```bash
python main_driver.py --version <VERSION> [--checkpoint <PATH_TO_CHECKPOINT>]
```

**Arguments:**
*   `--version`: Model architecture version.
    *   `1`: Centralized Actor-Critic (One Categorical Distribution).
    *   `2`: Centralized Actor-Critic (Agent-wise Distributions).
    *   `3` (or other): CoMapping Policy (Specialized GNN with Map Encoding).
*   `--checkpoint`: (Optional) Path to a `.pt` file to resume training.
*   `--map_type`: (Optional) Specific map type for testing during training (`corridor`, `maze`, `random`).

### Testing & Visualization

To evaluate a trained model and generate visualizations (GIFs):

```bash
python test_main_driver.py --checkpoint <PATH_TO_CHECKPOINT> --version <VERSION>
```

**Arguments:**
*   `--checkpoint`: **Required**. Path to the model checkpoint file.
*   `--version`: Model version used during training (`1`, `2`, or `3`).
*   `--map_type`: Map type to test on (`corridor`, `maze`, `random`, `single_maze`).
*   `--episodes`: Number of episodes to run (default: 10).
*   `--visualize`: Set to `True` to enable GIF generation for specific evaluation.

## Models

1.  **Version 1 (`RL_ActorCritic`)**: A standard centralized PPO architecture where a GNN processes agent observations, and a shared actor-critic network determines actions.
2.  **Version 2 (`RL_Policy`)**: Similar to Version 1 but utilizes agent-wise distributions for more granular control over individual agent actions.
3.  **Version 3 (`RL_CoMapping_Policy`)**: The default and most advanced model. It integrates a `MapEncoder` and `Differentiable Optimal Transport` (Sinkhorn) to better handle spatial reasoning and coordination tasks.

## Results