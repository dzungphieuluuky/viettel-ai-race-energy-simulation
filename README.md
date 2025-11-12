# RL-Based Energy Saving Agent for 5G Networks

This project implements a Reinforcement Learning (RL) agent for energy-saving and power control in 5G networks. The agent is trained using a Soft Actor-Critic (SAC) algorithm with curriculum learning to optimize energy efficiency while maintaining network performance.

---

## Features

- **Custom RL Environment**: Simulates a 5G network with user equipment (UEs) and base stations (cells).
- **Curriculum Learning**: Gradual introduction of objectives to stabilize training and improve performance.
- **Stabilized SAC**: Includes techniques like Huber loss, gradient clipping, and reward normalization for robust training.
- **Replay Buffer**: Stores experiences for off-policy training.
- **Energy Optimization**: Focuses on reducing energy consumption while ensuring network KPIs are met.
- **Logging**: Detailed logs for debugging and monitoring training progress.

---

## Project Structure
```bash!
├── app/
│ ├── energy_agent/
│ │ ├── rl_agent.py # Main RL agent implementation
│ │ ├── state_normalizer.py # State normalization utilities
│ └── simulation/
│ ├── utils/ # Helper functions for simulation
│ ├── simCore/ # Core simulation logic
│ └── run_simulation.py # Entry point for running simulations
├── online_trained_models_manual_sac/ # Directory for saving trained models
├── logs/ # Directory for log files
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Stable-Baselines3
- Gymnasium

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/energy-saving-agent.git
   cd energy-saving-agent
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Usage:
    ```bash
    Check the DockerUsage.md file.

## Key Components
### RL Agent (rl_agent.py)
- **Actor-Critic Architecture**: Implements the SAC algorithm with separate actor and critic networks.
- **Curriculum Learning**: Gradually increases the complexity of the reward function.
- **Replay Buffer**: Stores past experiences for off-policy training.
- **Reward Calculator**: Quite strict and complicated reward calculator to prevent reward hacking.
- **Constraint Penalties**: Penalizes violations of network KPIs (e.g., latency, drop rate).
- **Energy Rewards**: Rewards the agent for reducing energy consumption.
- **Normalization**: Stabilizes rewards for better training.
- **State Normalizer** (state_normalizer.py)
Normalizes state features to ensure consistent input to the neural networks.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- PyTorch
- Stable Baselines 3
- Gymnasium



