import os
import time
import json
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/data_collection.log"),
        logging.StreamHandler()
    ]
)

class BrowserDataCollector:
    """
    Collects browser interaction data for training the RL model.
    This class implements a semi-random exploration strategy to
    gather diverse browser interaction data.
    """
    
    def __init__(self, start_urls=None, max_steps=100, output_dir="data"):
        """
        Initialize the data collector.
        
        Args:
            start_urls (list): List of URLs to start browsing from
            max_steps (int): Maximum number of steps per browsing session
            output_dir (str): Directory to save collected data
        """
        if start_urls is None:
            self.start_urls = [
                "https://www.google.com",
                "https://www.wikipedia.org",
                "https://www.github.com",
                "https://www.reddit.com",
                "https://news.ycombinator.com"
            ]
        else:
            self.start_urls = start_urls
            
        self.max_steps = max_steps
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize counters and storage
        self.session_count = 0
        self.total_interactions = 0
        self.successful_interactions = 0
        self.data = []
        
        self.driver = None
    
    def start_collection(self, num_sessions=10):
        """
        Start collecting data for multiple browsing sessions.
        
        Args:
            num_sessions (int): Number of browsing sessions to run
        """
        logging.info(f"Starting data collection for {num_sessions} sessions")
        
        for i in range(num_sessions):
            try:
                logging.info(f"Starting session {i+1}/{num_sessions}")
                self._run_session()
                self.session_count += 1
                
                # Save data after each session
                self._save_data()
                
            except Exception as e:
                logging.error(f"Error in session {i+1}: {e}")
                # Close and restart the browser
                if self.driver:
                    self.driver.quit()
                    self.driver = None
        
        # Final statistics
        success_rate = (self.successful_interactions / self.total_interactions) * 100 if self.total_interactions > 0 else 0
        logging.info(f"Data collection completed. Sessions: {self.session_count}, "
                    f"Total interactions: {self.total_interactions}, "
                    f"Successful: {self.successful_interactions} ({success_rate:.2f}%)")
    
    def _init_browser(self):
        """Initialize the Selenium browser."""
        options = webdriver.ChromeOptions()
        # For data collection, we want to see the browser
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-notifications')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1024, 768)
    
    def _run_session(self):
        """Run a single browsing session."""
        if not self.driver:
            self._init_browser()
        
        # Start from a random URL
        start_url = random.choice(self.start_urls)
        self.driver.get(start_url)
        time.sleep(2)  # Wait for page to load
        
        for step in range(self.max_steps):
            try:
                # Get current state information
                current_state = self._get_current_state()
                
                # Choose a random action
                action_type, element = self._choose_action()
                
                # Execute the action and get result
                success, result_info = self._execute_action(action_type, element)
                
                # Record the interaction
                self._record_interaction(current_state, action_type, success, result_info)
                
                # Allow page to update
                time.sleep(1.5)
                
            except Exception as e:
                logging.warning(f"Error in step {step+1}: {e}")
                # Try to continue with the session
                continue
    
    def _get_current_state(self):
        """Get the current state of the browser."""
        try:
            # Capture basic page information
            state = {
                "url": self.driver.current_url,
                "title": self.driver.title,
                "timestamp": datetime.now().isoformat()
            }
            
            # Capture screenshot
            screenshot_path = os.path.join(
                self.output_dir, 
                f"screenshot_session_{self.session_count}_step_{len(self.data)}.png"
            )
            self.driver.save_screenshot(screenshot_path)
            state["screenshot_path"] = screenshot_path
            
            # Parse the page to get interactive elements
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Count interactive elements
            state["links_count"] = len(soup.find_all('a'))
            state["buttons_count"] = len(soup.find_all('button'))
            state["inputs_count"] = len(soup.find_all(['input', 'textarea']))
            state["selects_count"] = len(soup.find_all('select'))
            
            return state
            
        except Exception as e:
            logging.error(f"Error getting current state: {e}")
            return {"error": str(e)}
    
    def _choose_action(self):
        """
        Choose a random action to perform.
        
        Returns:
            tuple: (action_type, element) where element might be None for some actions
        """
        # Define possible action types with weights (higher = more likely)
        action_types = {
            "click": 40,
            "type": 25,
            "press_enter": 10,
            "back": 5,
            "forward": 5,
            "scroll": 15
        }
        
        # Choose action type based on weights
        action_choices = []
        action_weights = []
        
        for action, weight in action_types.items():
            action_choices.append(action)
            action_weights.append(weight)
        
        action_weights = np.array(action_weights) / sum(action_weights)
        action_type = np.random.choice(action_choices, p=action_weights)
        
        element = None
        
        # For click and type actions, we need to find an element
        if action_type == "click":
            element = self._find_random_clickable()
        elif action_type == "type":
            element = self._find_random_input()
        
        return action_type, element
    
    def _find_random_clickable(self):
        """Find a random clickable element on the page."""
        try:
            # Find all clickable elements
            clickable_elements = self.driver.find_elements(By.XPATH, 
                "//a | //button | //input[@type='button'] | //input[@type='submit'] | //div[@role='button']")
            
            if not clickable_elements:
                return None
            
            # Return a random element
            return random.choice(clickable_elements)
            
        except Exception as e:
            logging.warning(f"Error finding clickable element: {e}")
            return None
    
    def _find_random_input(self):
        """Find a random input element on the page."""
        try:
            # Find all input fields
            input_elements = self.driver.find_elements(By.XPATH, 
                "//input[@type='text'] | //textarea | //input[@type='search'] | //input[@type='email'] | //input[@type='password']")
            
            if not input_elements:
                return None
            
            # Return a random element
            return random.choice(input_elements)
            
        except Exception as e:
            logging.warning(f"Error finding input element: {e}")
            return None
    
    def _execute_action(self, action_type, element):
        """
        Execute the chosen action.
        
        Args:
            action_type (str): Type of action to perform
            element: Selenium element for the action (may be None)
            
        Returns:
            tuple: (success, result_info)
        """
        self.total_interactions += 1
        result_info = {"action_type": action_type}
        
        try:
            if action_type == "click" and element:
                # Scroll element into view
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                time.sleep(0.5)
                
                # Save element info before clicking
                result_info["element_tag"] = element.tag_name
                result_info["element_text"] = element.text[:100] if element.text else ""
                
                # Click the element
                element.click()
                
                # Record the URL after clicking
                time.sleep(1)  # Wait for possible navigation
                result_info["result_url"] = self.driver.current_url
                
                self.successful_interactions += 1
                return True, result_info
                
            elif action_type == "type" and element:
                # Scroll element into view
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                time.sleep(0.5)
                
                # Save element info before typing
                result_info["element_tag"] = element.tag_name
                
                # Clear existing text and type new text
                element.clear()
                
                # Generate random search queries
                search_queries = [
                    "latest news today",
                    "recipe for chocolate cake",
                    "how to learn programming",
                    "best movies of 2023",
                    "vacation destinations",
                    "how to fix a leaky faucet",
                    "Python programming tutorial",
                    "machine learning examples",
                    "reinforcement learning browser automation"
                ]
                
                search_text = random.choice(search_queries)
                element.send_keys(search_text)
                result_info["typed_text"] = search_text
                
                self.successful_interactions += 1
                return True, result_info
                
            elif action_type == "press_enter":
                # Find the active element
                active_element = self.driver.switch_to.active_element
                
                # Send Enter key
                active_element.send_keys(Keys.ENTER)
                time.sleep(1)  # Wait for possible navigation
                
                result_info["result_url"] = self.driver.current_url
                
                self.successful_interactions += 1
                return True, result_info
                
            elif action_type == "back":
                # Store current URL before going back
                result_info["from_url"] = self.driver.current_url
                
                # Go back
                self.driver.back()
                time.sleep(1)  # Wait for navigation
                
                result_info["to_url"] = self.driver.current_url
                
                self.successful_interactions += 1
                return True, result_info
                
            elif action_type == "forward":
                # Store current URL before going forward
                result_info["from_url"] = self.driver.current_url
                
                # Go forward
                self.driver.forward()
                time.sleep(1)  # Wait for navigation
                
                result_info["to_url"] = self.driver.current_url
                
                self.successful_interactions += 1
                return True, result_info
                
            elif action_type == "scroll":
                # Choose scroll direction and amount
                directions = ["up", "down"]
                direction = random.choice(directions)
                
                if direction == "up":
                    scroll_amount = random.randint(-500, -100)
                else:
                    scroll_amount = random.randint(100, 500)
                
                result_info["scroll_direction"] = direction
                result_info["scroll_amount"] = scroll_amount
                
                # Execute scroll
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                
                self.successful_interactions += 1
                return True, result_info
                
            else:
                return False, {"error": "Invalid action or missing element"}
                
        except Exception as e:
            logging.warning(f"Action execution error: {e}")
            result_info["error"] = str(e)
            return False, result_info
    
    def _record_interaction(self, state, action_type, success, result_info):
        """Record the interaction data."""
        interaction_data = {
            "session_id": self.session_count,
            "interaction_id": len(self.data),
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "action": {
                "type": action_type,
                "success": success,
                "details": result_info
            }
        }
        
        self.data.append(interaction_data)
    
    def _save_data(self):
        """Save the collected data to disk."""
        try:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"browser_data_session_{self.session_count}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the data as JSON
            with open(filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
                
            logging.info(f"Saved {len(self.data)} interactions to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving data: {e}")
    
    def close(self):
        """Close the browser and cleanup."""
        if self.driver:
            self.driver.quit()
            self.driver = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect browser interaction data for RL training")
    parser.add_argument("--sessions", type=int, default=5, help="Number of browsing sessions to run")
    parser.add_argument("--steps", type=int, default=50, help="Maximum steps per session")
    parser.add_argument("--output", type=str, default="data", help="Output directory for collected data")
    
    args = parser.parse_args()
    
    collector = BrowserDataCollector(
        max_steps=args.steps,
        output_dir=args.output
    )
    
    try:
        collector.start_collection(num_sessions=args.sessions)
    finally:
        collector.close()