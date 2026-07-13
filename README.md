# Flappy Bird AI using Deep Q-Network (DQN)

A Reinforcement Learning project that trains an AI agent to play **Flappy Bird** using the **Deep Q-Network (DQN)** algorithm implemented with **PyTorch** and the **Gymnasium Flappy Bird Environment**.

The agent learns by interacting with the environment, storing experiences in a replay buffer, and optimizing a neural network to maximize cumulative rewards.

---

## Project Overview

Unlike supervised learning, Reinforcement Learning enables an agent to learn through trial and error. In this project, a DQN agent is trained to master Flappy Bird by learning the optimal timing for flapping through continuous interaction with the game environment.

The project implements:

- Deep Q-Network (DQN)
- Experience Replay
- Target Network Synchronization
- Epsilon-Greedy Exploration
- Reward-Based Learning
- Model Saving and Loading

---

## Features

- Custom DQN implementation in PyTorch
- Experience Replay Memory
- Target Network for stable learning
- Configurable hyperparameters using YAML
- Automatic model checkpointing
- Training and inference modes
- GPU (CUDA/MPS) support
- Play trained model in real-time

---

## Project Structure

```
.
├── fp.py                  # Main training & testing script
├── dqn.py                 # Deep Q-Network architecture
├── experience_replay.py   # Replay Buffer implementation
├── flappy.py              # Manual Flappy Bird gameplay
├── parameters.yaml        # Hyperparameters
├── runs/
│   ├── model.pt
│   └── training.log
├── README.md
```

---

## Model Architecture

The Deep Q-Network consists of:

```
Input Layer (State Vector)
        │
        ▼
Fully Connected Layer (256 Neurons)
        │
      ReLU
        │
        ▼
Output Layer
(Q-values for each action)
```

Default Network Configuration:

- Input Features: **12**
- Hidden Units: **256**
- Output Actions:
  - 0 → Do Nothing
  - 1 → Flap

---

## Environment

Environment used:

- Gymnasium
- Flappy Bird Gymnasium

Observation Space

- 12-dimensional state vector

Action Space

| Action | Description |
|--------|-------------|
| 0 | Do Nothing |
| 1 | Flap |

---

## Training Pipeline

```
Environment
      │
      ▼
Observe State
      │
      ▼
Choose Action
(Epsilon-Greedy)
      │
      ▼
Perform Action
      │
      ▼
Receive Reward
      │
      ▼
Store Experience
      │
      ▼
Sample Mini-Batch
      │
      ▼
Train DQN
      │
      ▼
Update Target Network
```

---

## Reinforcement Learning Components

### Experience Replay

Experiences are stored in a replay buffer:

```
(State,
 Action,
 Next State,
 Reward,
 Done)
```

Random mini-batches are sampled to reduce correlation between consecutive experiences.

---

### Epsilon-Greedy Strategy

The agent balances:

- Exploration (random actions)
- Exploitation (best predicted action)

Epsilon gradually decays during training to encourage learning.

---

### Target Network

A separate target network is synchronized periodically with the policy network to stabilize training.

---

## Technologies Used

- Python
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- NumPy
- PyYAML

---

## Installation

Install dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

---

## Training

Train the DQN agent:

```bash
python fp.py default --train
```

---

## Test the Trained Agent

```bash
python fp.py default
```

The trained agent will automatically load the saved model and play Flappy Bird.

---

## Manual Gameplay

You can also play manually:

```bash
python flappy.py
```

Controls:

- **Spacebar** → Flap

---

## Training Results

During training, the console displays:

```
Episode : 150
Reward : 42
Epsilon : 0.31
```

The best-performing model is automatically saved in the `runs/` directory whenever a higher reward is achieved.

---

## Results

- Successfully implemented Deep Q-Learning
- Experience Replay improves training stability
- Target Network reduces Q-value oscillations
- Automatic model checkpointing
- Trained agent learns to survive progressively longer by maximizing cumulative reward
- Supports CPU, CUDA, and Apple Silicon (MPS)

---

## Future Improvements

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Rainbow DQN
- TensorBoard integration
- Training performance graphs
- Hyperparameter optimization

---

## Learning Outcomes

This project demonstrates:

- Reinforcement Learning fundamentals
- Deep Q-Learning
- Neural Networks with PyTorch
- Experience Replay
- Target Networks
- Epsilon-Greedy Exploration
- Sequential decision making

---

## License

This project is intended for educational and research purposes.

---
