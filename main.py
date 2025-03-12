#!/usr/bin/env python3
"""
Main script for the Browser Reinforcement Learning project.
This script provides a command-line interface to run different parts of the project.
"""

import os
import argparse
import logging
import time

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

def setup_env():
    """Run environment setup."""
    from setup import install_requirements, setup_browser_drivers
    
    logging.info("Setting up environment...")
    install_requirements()
    setup_browser_drivers()
    
    # Create project directories
    directories = ["data", "models", "logs", "configs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    logging.info("Environment setup completed")

def collect_data(args):
    """Run data collection."""
    from data_collector import BrowserDataCollector
    
    logging.info("Starting data collection...")
    
    collector = BrowserDataCollector(
        max_steps=args.steps,
        output_dir=args.output_dir
    )
    
    try:
        collector.start_collection(num_sessions=args.sessions)
    finally:
        collector.close()
    
    logging.info("Data collection completed")

def process_data(args):
    """Run data processing."""
    from data_processor import BrowserDataProcessor
    
    logging.info("Starting data processing...")
    
    processor = BrowserDataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Process the data
    num_loaded = processor.load_data()
    
    if num_loaded > 0:
        processor.preprocess_data()
        processor.extract_visual_features(batch_size=args.batch_size)
        processor.create_training_dataset()
    else:
        logging.error("No data loaded. Please collect data first.")
    
    logging.info("Data processing completed")

def train_model(args):
    """Train the RL model."""
    from rl_model import BrowserRL
    
    logging.info("Starting model training...")
    
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
    
    logging.info("Model training completed")

def run_agent(args):
    """Run the trained agent."""
    from browser_agent import BrowserAgent
    
    logging.info("Starting browser agent...")
    
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
    
    logging.info("Browser agent run completed")

def main():
    parser = argparse.ArgumentParser(
        description="Browser Reinforcement Learning Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the environment")
    
    # Data collection command
    collect_parser = subparsers.add_parser("collect", help="Collect browser interaction data")
    collect_parser.add_argument("--sessions", type=int, default=5, help="Number of browsing sessions to run")
    collect_parser.add_argument("--steps", type=int, default=50, help="Maximum steps per session")
    collect_parser.add_argument("--output-dir", type=str, default="data", help="Output directory for collected data")
    
    # Data processing command
    process_parser = subparsers.add_parser("process", help="Process collected data")
    process_parser.add_argument("--data-dir", type=str, default="data", help="Directory containing collected data")
    process_parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed data")
    process_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the RL model")
    train_parser.add_argument("--algorithm", type=str, choices=["PPO", "A2C", "DQN"], default="PPO", 
                            help="RL algorithm to use")
    train_parser.add_argument("--policy", type=str, default="CnnPolicy", help="Policy network architecture")
    train_parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train for")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency during training")
    train_parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    train_parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models")
    train_parser.add_argument("--start-url", type=str, default="https://www.google.com", help="Starting URL")
    
    # Run agent command
    run_parser = subparsers.add_parser("run", help="Run the trained agent")
    run_parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    run_parser.add_argument("--algorithm", type=str, choices=["PPO", "A2C", "DQN"], default="PPO", help="RL algorithm used")
    run_parser.add_argument("--steps", type=int, default=100, help="Maximum number of steps to run")
    run_parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    run_parser.add_argument("--url", type=str, default="https://www.google.com", help="Starting URL")
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "setup":
        setup_env()
    elif args.command == "collect":
        collect_data(args)
    elif args.command == "process":
        process_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "run":
        run_agent(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()