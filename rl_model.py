import os
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our custom environment
from browser_env import BrowserEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/rl_training.log"),
        logging.StreamHandler()
    ]
)

class BrowserRL:
    """
    Reinforcement Learning model trainer for browser automation.
    Uses Stable Baselines3 implementations of popular RL algorithms.
    """
    
    def __init__(self, 
                 env_config=None,
                 algorithm="PPO",
                 policy="CnnPolicy",
                 models_dir="models"):
        """
        Initialize the RL trainer.
        
        Args:
            env_config (dict): Configuration for the browser environment
            algorithm (str): RL algorithm to use (PPO, A2C, DQN)
            policy (str): Policy network architecture
            models_dir (str): Directory to save models
        """
        self.env_config = env_config or {
            "start_url": "https://www.google.com",
            "headless": True,
            "time_limit": 500
        }
        
        self.algorithm = algorithm
        self.policy = policy
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize environment and model
        self.env = None
        self.model = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_environment(self):
        """Create and configure the browser environment."""
        logging.info("Creating browser environment...")
        
        # Create the base environment
        env = BrowserEnv(**self.env_config)
        
        # Wrap the environment for monitoring
        log_dir = os.path.join(self.models_dir, "monitor")
        os.makedirs(log_dir, exist_ok=True)
        
        env = Monitor(env, log_dir)
        
        # Vectorize the environment (required by Stable Baselines)
        env = DummyVecEnv([lambda: env])
        
        # Stack frames for temporal information
        env = VecFrameStack(env, n_stack=4)
        
        self.env = env
        logging.info("Environment created successfully")
        
        return env
    
    def create_model(self):
        """Create and configure the RL model."""
        logging.info(f"Creating {self.algorithm} model with {self.policy}...")
        
        if self.env is None:
            self.create_environment()
        
        # Create the appropriate algorithm
        if self.algorithm == "PPO":
            self.model = PPO(
                self.policy,
                self.env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                tensorboard_log=os.path.join(self.models_dir, "tensorboard_logs")
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                self.policy,
                self.env,
                verbose=1,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                normalize_advantage=False,
                tensorboard_log=os.path.join(self.models_dir, "tensorboard_logs")
            )
        elif self.algorithm == "DQN":
            self.model = DQN(
                self.policy,
                self.env,
                verbose=1,
                learning_rate=0.0004,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                tensorboard_log=os.path.join(self.models_dir, "tensorboard_logs")
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Use 'PPO', 'A2C', or 'DQN'.")
        
        logging.info(f"Model created successfully: {self.algorithm} with {self.policy}")
        
        return self.model
    
    def train(self, total_timesteps=100000, eval_freq=10000, n_eval_episodes=5):
        """
        Train the reinforcement learning model.
        
        Args:
            total_timesteps (int): Total number of timesteps to train for
            eval_freq (int): Frequency of evaluation during training
            n_eval_episodes (int): Number of episodes for evaluation
        """
        if self.model is None:
            self.create_model()
        
        logging.info(f"Starting training for {total_timesteps} timesteps...")
        
        # Setup callbacks
        checkpoint_dir = os.path.join(self.models_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{self.algorithm}_browser"
        )
        
        # Create a separate environment for evaluation
        eval_env = BrowserEnv(**self.env_config)
        eval_env = Monitor(eval_env, os.path.join(self.models_dir, "eval_monitor"))
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.models_dir, "best_model"),
            log_path=os.path.join(self.models_dir, "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Start training
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
        
        # Save the final model
        final_model_path = os.path.join(self.models_dir, f"final_{self.algorithm}_browser")
        self.model.save(final_model_path)
        logging.info(f"Training completed. Final model saved to {final_model_path}")
        
        return final_model_path
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        
        Returns:
            The loaded model
        """
        if not os.path.exists(model_path):
            logging.error(f"Model not found at {model_path}")
            return None
        
        logging.info(f"Loading model from {model_path}...")
        
        if self.algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif self.algorithm == "A2C":
            self.model = A2C.load(model_path)
        elif self.algorithm == "DQN":
            self.model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        logging.info("Model loaded successfully")
        
        return self.model
    
    def evaluate(self, n_eval_episodes=10, deterministic=True):
        """
        Evaluate the trained model.
        
        Args:
            n_eval_episodes (int): Number of episodes to evaluate
            deterministic (bool): Whether to use deterministic actions
        
        Returns:
            tuple: (mean_reward, std_reward)
        """
        if self.model is None:
            logging.error("No model loaded. Call create_model() or load_model() first.")
            return None, None
        
        logging.info(f"Evaluating model for {n_eval_episodes} episodes...")
        
        # Create a clean environment for evaluation
        eval_env = BrowserEnv(**self.env_config)
        eval_env = Monitor(eval_env, os.path.join(self.models_dir, "final_eval"))
        
        # Run evaluation
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic
        )
        
        logging.info(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model for browser automation")
    parser.add_argument("--algorithm", type=str, choices=["PPO", "A2C", "DQN"], default="PPO", 
                        help="RL algorithm to use")
    parser.add_argument("--policy", type=str, default="CnnPolicy", 
                        help="Policy network architecture")
    parser.add_argument("--timesteps", type=int, default=100000, 
                        help="Total timesteps to train for")
    parser.add_argument("--eval-freq", type=int, default=10000, 
                        help="Evaluation frequency during training")
    parser.add_argument("--headless", action="store_true", 
                        help="Run browser in headless mode")
    parser.add_argument("--models-dir", type=str, default="models", 
                        help="Directory to save models")
    parser.add_argument("--start-url", type=str, default="https://www.google.com", 
                        help="Starting URL for the browser environment")
    
    args = parser.parse_args()
    
    # Configure environment
    env_config = {
        "start_url": args.start_url,
        "headless": args.headless,
        "time_limit": 500
    }
    
    # Create and train the model
    rl_trainer = BrowserRL(
        env_config=env_config,
        algorithm=args.algorithm,
        policy=args.policy,
        models_dir=args.models_dir
    )
    
    try:
        rl_trainer.create_environment()
        rl_trainer.create_model()
        rl_trainer.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq
        )
        rl_trainer.evaluate()
    finally:
        # Clean up environment
        if rl_trainer.env:
            rl_trainer.env.close()