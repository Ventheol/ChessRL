# ChessRL ♟️

A reinforcement learning chess agent built from scratch using **PyTorch**.

This project explores how reinforcement learning can be applied to chess by implementing the full pipeline: environment design, neural network architecture, PPO training, self-play, and search experiments.

The goal of this project is not to create a world-class chess engine, but to gain practical experience with reinforcement learning engineering and deep learning system design.

🚧 **This project is under active development.**

---

# Motivation

Reinforcement learning is a field where understanding comes from building and experimenting.

Instead of only studying RL algorithms theoretically, this project was created to explore:

* How an agent learns from interaction
* How policy and value networks work together
* How self-play can improve decision making
* How search methods like MCTS can complement neural networks

This project is also a portfolio project demonstrating practical AI engineering skills.

---

# Features

## Reinforcement Learning Pipeline

Implemented a PPO-based reinforcement learning pipeline including:

* Proximal Policy Optimization (PPO)
* Actor-Critic architecture
* Rollout buffer
* Generalized Advantage Estimation (GAE)
* Policy optimization with clipped objective
* Self-play training support

---

## Training Approaches

The project currently uses two training stages.

## 1. Supervised Pre-training

File:

```
pre_train.py
```

Before reinforcement learning, the model can be trained using chess game data.

Purpose:

* Give the model an initial understanding of chess positions
* Provide a stronger starting point before reinforcement learning

Training data is based on chess game records.

---

## 2. Reinforcement Learning Training

File:

```
train.py
```

The agent learns through interaction using PPO.

The agent supports:

* Self-play training
* Learning from generated experience
* Policy and value optimization

The model can also be played against through:

```
play.py
```

---

# Neural Network Architecture

The agent uses a convolutional neural network inspired by architectures commonly used in modern board-game AI.

Architecture:

```
Chess Board Tensor
        |
        v
Convolution Layers
        |
        v
Batch Normalization + ReLU
        |
        v
Residual Blocks (4x)
        |
        v
Feature Projection (256)
        |
        +----------------+
        |                |
        v                v
   Policy Head       Value Head
    (Actor)          (Critic)
```

The network contains:

* CNN layers for spatial board understanding
* BatchNorm2d layers
* ReLU activation functions
* Four residual blocks
* Actor head for move probabilities
* Critic head for board evaluation

---

# Chess Board Representation

The chess state is represented as:

```
8 x 8 x 18
```

tensor with `float32` values.

The board representation uses multiple feature planes to encode chess information and is processed by convolution layers.

Implementation:

```
chess_rl/env/chess_env.py
```

---

# Legal Move Masking

One of the first challenges during development was that an untrained neural network does not understand chess rules.

Early experiments showed the model selecting impossible actions, such as attempting illegal piece movements.

To solve this, legal move masking was implemented.

The policy output is filtered so the agent can only select moves that are valid in the current chess position.

This improves training stability and prevents the model from wasting experience on impossible actions.

---

# MCTS Experiment

During development, an experimental Monte Carlo Tree Search (MCTS) module was implemented.

The motivation was to explore whether tree-based search could help the neural network avoid obvious mistakes by looking ahead at possible future positions.

Current status:

* MCTS prototype implemented
* Used as an experimental search component
* Further integration and improvement are planned

Implementation:

```
chess_rl/mcts/
```

---

# Project Structure

```
ChessRL/
│
├── train.py                  # PPO training
├── pre_train.py              # Supervised pre-training
├── play.py                   # Play against the agent
│
├── chess_rl/
│   │
│   ├── env/                  # Chess environment
│   │
│   ├── models/               # Neural network architecture
│   │
│   ├── ppo/                  # PPO implementation
│   │
│   ├── mcts/                 # MCTS experiments
│   │
│   └── tests/                # Unit tests
│
├── requirements.txt
└── README.md
```

---

# Technologies

Built with:

* Python
* PyTorch
* NumPy
* python-chess

---

# Current Progress

Implemented:

✅ Custom chess environment
✅ Chess board tensor representation
✅ CNN + Residual network architecture
✅ Actor-Critic model
✅ PPO training pipeline
✅ GAE advantage calculation
✅ Rollout buffer
✅ Legal move masking
✅ Self-play support
✅ Human vs AI gameplay
✅ Experimental MCTS implementation

---

# Future Improvements

Planned improvements:

* Add evaluation metrics
* Measure playing strength
* Improve self-play stability
* Tune PPO hyperparameters
* Improve MCTS integration
* Compare different training strategies
* Create better gameplay visualization

---

# Lessons Learned

Building this project helped explore practical challenges in reinforcement learning:

* Designing useful state representations
* Handling invalid actions
* Debugging unstable RL training
* Understanding policy/value networks
* Combining learning-based methods with search algorithms

The project is an ongoing experiment in understanding how intelligent agents can learn complex decision-making problems.

---

# License

This project is released under the MIT License.

```
```
