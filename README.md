# CasualSurfer

Casual surfing companion powered by AI.

## Overview

This project implements a deep reinforcement learning (DRL) system that learns to interact with web browsers, performing tasks like clicking links, filling forms, scrolling pages, and navigating between websites. The agent perceives web content through screenshots and learns optimal actions through reinforcement learning.

## Key Features

- **Browser Environment**: Custom Gymnasium environment for browser interaction using Selenium
- **Data Collection**: Automated system to gather browser interaction data
- **Feature Extraction**: Processes raw browser screenshots into ML-ready features
- **Reinforcement Learning**: Implements PPO, A2C, and DQN algorithms for training
- **Deployment**: Ready-to-use agent that can navigate websites autonomously

## Installation

### Prerequisites

- Python 3.8+
- Chrome browser installed on your system

### Setup

1. Clone the repository:

```bash
git clone https://github.com/valekar/casual-surfer.git
cd browser-rl
```

2. Run the setup script:

```bash
python main.py setup
```

This will:

- Install all required packages from `requirements.txt`
- Set up Chrome WebDriver for Selenium
- Create necessary project directories

## Project Structure

```
browser-rl/
├── browser_env.py        # Browser environment (Gymnasium)
├── data_collector.py     # Data collection module
├── data_processor.py     # Data preprocessing and feature extraction
├── rl_model.py           # RL model definition and training
├── browser_agent.py      # Deployment agent for trained models
├── main.py               # Command-line interface
├── setup.py              # Environment setup script
├── requirements.txt      # Required Python packages
├── data/                 # Directory for collected data
├── models/               # Directory for trained models
├── logs/                 # Directory for log files
└── configs/              # Directory for configuration files
```

## Step-by-Step Usage Guide

### 1. Data Collection

Collect browser interaction data for training:

```bash
python main.py collect --sessions 10 --steps 100
```

Options:

- `--sessions`: Number of browsing sessions to run
- `--steps`: Maximum number of steps per session
- `--output-dir`: Directory to save collected data

### 2. Data Processing

Process the collected data to create training features:

```bash
python main.py process --data-dir data --output-dir data/processed
```

Options:

- `--data-dir`: Directory containing collected data
- `--output-dir`: Directory to save processed data
- `--batch-size`: Batch size for feature extraction

### 3. Model Training

Train the reinforcement learning model:

```bash
python main.py train --algorithm PPO --timesteps 100000 --headless
```

Options:

- `--algorithm`: RL algorithm to use (PPO, A2C, DQN)
- `--policy`: Policy network architecture
- `--timesteps`: Total timesteps to train for
- `--eval-freq`: Frequency of evaluation during training
- `--headless`: Run browser in headless mode
- `--models-dir`: Directory to save models
- `--start-url`: Starting URL for the browser environment

### 4. Running the Trained Agent

Run the trained agent on real websites:

```bash
python main.py run --model models/final_PPO_browser.zip --steps 200
```

Options:

- `--model`: Path to the trained model file
- `--algorithm`: RL algorithm used
- `--steps`: Maximum number of steps to run
- `--headless`: Run browser in headless mode
- `--url`: Starting URL

## Technical Architecture

### Browser Environment

The browser environment (`BrowserEnv`) implements the Gymnasium interface and provides:

- **Observation Space**: Browser screenshots resized to 84x84 pixels
- **Action Space**: Discrete actions (click, type, press enter, back, forward, scroll)
- **Reward Function**: Rewards based on successful interactions and discoveries

### Reinforcement Learning Model

The system supports three RL algorithms:

- **PPO (Proximal Policy Optimization)**: Balances exploration and exploitation
- **A2C (Advantage Actor-Critic)**: Lower sample complexity but less stable
- **DQN (Deep Q-Network)**: Good for discrete action spaces

### Data Pipeline

1. **Collection**: Semi-random exploration of websites
2. **Processing**: Image preprocessing and feature extraction
3. **Training**: End-to-end RL training using stable-baselines3
4. **Deployment**: Browser agent for interacting with real websites

## Customization

### Training on Specific Websites

To train on specific websites, modify the `start_urls` list in `data_collector.py`:

```python
self.start_urls = [
    "https://www.example1.com",
    "https://www.example2.com",
    # Add your websites here
]
```

### Modifying Reward Function

The reward function can be customized in `browser_env.py` by modifying the reward calculations in the action methods.

### Custom Policies

You can use different neural network architectures by specifying the policy parameter:

```bash
python main.py train --policy MlpPolicy  # For non-image observations
```

## Troubleshooting

### Chrome Driver Issues

If you encounter issues with Chrome WebDriver:

1. Verify Chrome is installed and up to date
2. Try manually installing the correct chromedriver version from https://chromedriver.chromium.org/downloads
3. Set the path environment variable to the chromedriver location

### Training Stability Issues

If training is unstable:

1. Reduce learning rate: Modify the `learning_rate` parameter in `rl_model.py`
2. Increase batch size: Use `--batch-size 64` or higher
3. Try different algorithms: A2C might be more stable than PPO for certain tasks

## Performance Optimization

### For Faster Training

- Use headless mode: Add `--headless` flag during training
- Reduce observation size: Modify `observation_height` and `observation_width` in `browser_env.py`
- Use GPU acceleration: Ensure TensorFlow/PyTorch is configured for GPU

### For Better Results

- Increase training steps: Use `--timesteps 500000` or higher
- Collect more diverse data: Increase `--sessions` during data collection
- Fine-tune hyperparameters: Adjust learning rates and batch sizes in `rl_model.py`

## Limitations and Future Work

- The current implementation handles only basic browser interactions
- Performance varies across different types of websites
- No support yet for complex interactions like drag-and-drop
- Future work could include:
  - Integration with NLP for text understanding
  - Task-specific fine-tuning
  - Multi-task learning across different websites

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL implementations
- Browser interaction is powered by [Selenium](https://www.selenium.dev/)
- Environment design follows [Gymnasium](https://gymnasium.farama.org/) interface
