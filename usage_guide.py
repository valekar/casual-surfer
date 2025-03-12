#!/usr/bin/env python3
"""
Usage guide for the Browser Automation with Deep Reinforcement Learning project.
This script demonstrates how to use the system with concrete examples.
"""

import os
import argparse
import json
import logging
import time
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_command(cmd):
    """Print a command with proper formatting."""
    print(f"\n$ {cmd}\n")

def run_command(cmd, show_output=True):
    """Run a shell command and optionally display its output."""
    print_command(cmd)
    
    try:
        if show_output:
            # Run command and show output in real-time
            process = subprocess.Popen(
                cmd, 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line.strip())
                
            process.wait()
            return process.returncode
        else:
            # Run command without showing output
            return subprocess.call(cmd, shell=True)
            
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return 1

def demo_setup():
    """Demonstrate environment setup."""
    print_header("1. ENVIRONMENT SETUP")
    print("""
This step installs all required dependencies and sets up the Chrome WebDriver.
It also creates necessary project directories.
    """)
    
    run_command("python main.py setup")

def demo_data_collection():
    """Demonstrate data collection."""
    print_header("2. DATA COLLECTION")
    print("""
This step collects browser interaction data for training the RL model.
It simulates random browsing sessions and records states, actions, and rewards.
    """)
    
    print("Example 1: Collect data with default settings (5 sessions, 50 steps each)")
    run_command("python main.py collect")
    
    print("\nExample 2: Collect more data with custom parameters")
    run_command("python main.py collect --sessions 2 --steps 20")

def demo_data_processing():
    """Demonstrate data processing."""
    print_header("3. DATA PROCESSING")
    print("""
This step processes the collected data to create features for training.
It extracts visual features from screenshots and normalizes numerical data.
    """)
    
    run_command("python main.py process")

def demo_model_training():
    """Demonstrate model training."""
    print_header("4. MODEL TRAINING")
    print("""
This step trains the reinforcement learning model using the processed data.
It uses the Stable Baselines3 implementation of RL algorithms.
    """)
    
    print("Example 1: Train a PPO model with default settings")
    print("Note: We'll use a very small number of timesteps for demonstration purposes.")
    run_command("python main.py train --algorithm PPO --timesteps 1000 --eval-freq 500 --headless")
    
    print("\nExample 2: Train an A2C model with different parameters")
    print("python main.py train --algorithm A2C --timesteps 1000 --eval-freq 500 --headless")

def demo_agent_running():
    """Demonstrate running the trained agent."""
    print_header("5. RUNNING THE TRAINED AGENT")
    print("""
This step runs the trained agent to automate browser interactions.
The agent will navigate websites and interact with elements autonomously.
    """)
    
    # Check if any models exist
    model_path = os.path.join("models", "final_PPO_browser.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "checkpoints", "PPO_browser_500_steps.zip")
    
    if os.path.exists(model_path):
        run_command(f"python main.py run --model {model_path} --steps 20")
    else:
        print("""
No trained models found. You need to train a model first using:
python main.py train --algorithm PPO --timesteps 1000 --eval-freq 500 --headless
        """)

def demo_customization():
    """Demonstrate customization options."""
    print_header("6. CUSTOMIZATION OPTIONS")
    print("""
This system can be customized in many ways by modifying the configuration files
or passing command-line arguments.
    """)
    
    # Show configuration file
    print("Default configuration file (configs/default_config.json):")
    try:
        with open("configs/default_config.json", "r") as f:
            config = json.load(f)
            print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Error loading config file: {e}")
    
    print("""
You can create custom configuration files and use them with the --config parameter:
python main.py train --config configs/my_custom_config.json
    """)

def demo_advanced_usage():
    """Demonstrate advanced usage."""
    print_header("7. ADVANCED USAGE EXAMPLES")
    print("""
Here are some advanced examples of how to use this system:
    """)
    
    print("Example 1: Collect data from specific websites")
    print("""
# Modify data_collector.py to use specific websites:
self.start_urls = [
    "https://www.example.com",
    "https://www.yourwebsite.com"
]
    """)
    
    print("\nExample 2: Fine-tune a pre-trained model")
    print("""
# Load a pre-trained model and continue training:
python main.py train --algorithm PPO --timesteps 10000 --model models/pre_trained_model.zip
    """)
    
    print("\nExample 3: Running the agent with a specific task")
    print("""
# Modify the reward function in browser_env.py to reward completing a specific task
# Then run the agent:
python main.py run --model models/task_specific_model.zip --url https://www.example.com/task-page
    """)

def main():
    parser = argparse.ArgumentParser(
        description="Browser RL Usage Guide - Interactive Examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--setup", action="store_true", help="Demo environment setup")
    parser.add_argument("--collect", action="store_true", help="Demo data collection")
    parser.add_argument("--process", action="store_true", help="Demo data processing")
    parser.add_argument("--train", action="store_true", help="Demo model training")
    parser.add_argument("--run", action="store_true", help="Demo running the agent")
    parser.add_argument("--customize", action="store_true", help="Demo customization options")
    parser.add_argument("--advanced", action="store_true", help="Demo advanced usage")
    
    args = parser.parse_args()
    
    # If no specific demo is selected, show all
    if not any(vars(args).values()):
        args.all = True
    
    print_header("BROWSER AUTOMATION WITH DEEP REINFORCEMENT LEARNING")
    print("""
This guide demonstrates how to use the Browser RL system with practical examples.
Each section shows a different aspect of the system with runnable commands.
    """)
    
    if args.all or args.setup:
        demo_setup()
        
    if args.all or args.collect:
        demo_data_collection()
        
    if args.all or args.process:
        demo_data_processing()
        
    if args.all or args.train:
        demo_model_training()
        
    if args.all or args.run:
        demo_agent_running()
        
    if args.all or args.customize:
        demo_customization()
        
    if args.all or args.advanced:
        demo_advanced_usage()
    
    print_header("USAGE GUIDE COMPLETE")
    print("""
For more information, please refer to the README.md file.
If you have any questions or issues, please open an issue on GitHub.
    """)

if __name__ == "__main__":
    main()