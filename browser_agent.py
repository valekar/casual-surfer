import os
import time
import argparse
import logging
import numpy as np
from stable_baselines3 import PPO, A2C, DQN

# Import our custom environment
from browser_env import BrowserEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/browser_agent.log"),
        logging.StreamHandler()
    ]
)

class BrowserAgent:
    """
    A browser automation agent that uses a trained reinforcement learning model
    to interact with web browsers on behalf of users.
    """
    
    def __init__(self, model_path, algorithm="PPO", headless=False):
        """
        Initialize the browser agent.
        
        Args:
            model_path (str): Path to the trained model file
            algorithm (str): RL algorithm used (PPO, A2C, DQN)
            headless (bool): Whether to run the browser in headless mode
        """
        self.model_path = model_path
        self.algorithm = algorithm
        self.headless = headless
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Initialize environment and model
        self.env = None
        self.model = None
    
    def load_model(self):
        """Load the trained model."""
        logging.info(f"Loading {self.algorithm} model from {self.model_path}...")
        
        try:
            if self.algorithm == "PPO":
                self.model = PPO.load(self.model_path)
            elif self.algorithm == "A2C":
                self.model = A2C.load(self.model_path)
            elif self.algorithm == "DQN":
                self.model = DQN.load(self.model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            logging.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def initialize_environment(self, start_url="https://www.google.com", time_limit=1000):
        """
        Initialize the browser environment.
        
        Args:
            start_url (str): URL to start browsing from
            time_limit (int): Maximum number of steps
        """
        logging.info(f"Initializing browser environment with URL: {start_url}")
        
        try:
            self.env = BrowserEnv(
                start_url=start_url,
                headless=self.headless,
                time_limit=time_limit,
                render_mode="human"  # Show the browser window
            )
            
            # Reset the environment to get initial observation
            observation, info = self.env.reset()
            logging.info(f"Environment initialized. Current URL: {info.get('url', 'unknown')}")
            
            return observation, info
            
        except Exception as e:
            logging.error(f"Error initializing environment: {e}")
            return None, None
    
    def run(self, max_steps=100, start_url="https://www.google.com", time_limit=1000):
        """
        Run the browser agent to perform automated browsing.
        
        Args:
            max_steps (int): Maximum number of steps to run
            start_url (str): URL to start browsing from
            time_limit (int): Maximum allowed time steps
        
        Returns:
            dict: Statistics about the run
        """
        # Load model if not already loaded
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": "Failed to load model"}
        
        # Initialize environment if not already initialized
        if self.env is None:
            observation, info = self.initialize_environment(start_url, time_limit)
            if observation is None:
                return {"error": "Failed to initialize environment"}
        else:
            observation, info = self.env.reset()
        
        # Statistics to track
        stats = {
            "steps": 0,
            "total_reward": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "visited_urls": set(),
            "action_distribution": {i: 0 for i in range(self.env.action_space.n)}
        }
        
        # Add starting URL to visited URLs
        stats["visited_urls"].add(info.get("url", "unknown"))
        
        logging.info(f"Starting automated browsing for {max_steps} steps")
        
        # Main loop
        for step in range(max_steps):
            try:
                # Get action from model
                action, _states = self.model.predict(observation, deterministic=True)
                
                # Execute the action
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Update statistics
                stats["steps"] += 1
                stats["total_reward"] += reward
                stats["action_distribution"][action] += 1
                
                if reward > 0:
                    stats["successful_actions"] += 1
                else:
                    stats["failed_actions"] += 1
                
                # Add URL to visited URLs
                stats["visited_urls"].add(info.get("url", "unknown"))
                
                # Log progress
                if step % 10 == 0 or reward > 0.5:
                    logging.info(f"Step {step}/{max_steps}, Action: {action}, Reward: {reward:.2f}, "
                                f"URL: {info.get('url', 'unknown')}")
                
                # Check if episode is done
                if terminated or truncated:
                    logging.info(f"Episode ended after {step+1} steps")
                    break
                
                # Small delay for stability and to observe the browser
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error in step {step}: {e}")
                stats["error"] = str(e)
                break
        
        # Convert visited URLs set to list for reporting
        stats["visited_urls"] = list(stats["visited_urls"])
        stats["unique_urls_visited"] = len(stats["visited_urls"])
        
        # Calculate success rate
        total_actions = stats["successful_actions"] + stats["failed_actions"]
        stats["success_rate"] = (stats["successful_actions"] / total_actions * 100) if total_actions > 0 else 0
        
        logging.info(f"Automated browsing completed. Steps: {stats['steps']}, "
                    f"Total reward: {stats['total_reward']:.2f}, "
                    f"Success rate: {stats['success_rate']:.2f}%")
        
        return stats
    
    def close(self):
        """Close the environment and release resources."""
        if self.env:
            self.env.close()
            logging.info("Environment closed")


def main():
    parser = argparse.ArgumentParser(description="Run a trained browser automation agent")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the trained model file")
    parser.add_argument("--algorithm", type=str, choices=["PPO", "A2C", "DQN"], default="PPO", 
                        help="RL algorithm used")
    parser.add_argument("--steps", type=int, default=100, 
                        help="Maximum number of steps to run")
    parser.add_argument("--headless", action="store_true", 
                        help="Run browser in headless mode")
    parser.add_argument("--url", type=str, default="https://www.google.com", 
                        help="Starting URL")
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = BrowserAgent(
        model_path=args.model,
        algorithm=args.algorithm,
        headless=args.headless
    )
    
    try:
        stats = agent.run(
            max_steps=args.steps,
            start_url=args.url
        )
        
        # Print summary
        print("\n===== BROWSER AGENT SUMMARY =====")
        print(f"Steps completed: {stats['steps']}")
        print(f"Total reward: {stats['total_reward']:.2f}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        print(f"Unique URLs visited: {stats['unique_urls_visited']}")
        print("\nAction distribution:")
        for action, count in stats["action_distribution"].items():
            if count > 0:
                print(f"  Action {action}: {count} times")
        
    finally:
        agent.close()


if __name__ == "__main__":
    main()