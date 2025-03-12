import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from bs4 import BeautifulSoup
import logging
import os
from PIL import Image
import io
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/browser_env.log"),
        logging.StreamHandler()
    ]
)

class BrowserEnv(gym.Env):
    """
    A Gymnasium environment for browser interaction using Selenium.
    
    This environment allows a reinforcement learning agent to interact
    with web browsers by observing the page state and taking actions
    like clicking, typing, scrolling, etc.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 start_url="https://www.google.com", 
                 headless=False, 
                 time_limit=500,
                 render_mode='rgb_array',
                 observation_height=84,
                 observation_width=84):
        super(BrowserEnv, self).__init__()
        
        self.start_url = start_url
        self.headless = headless
        self.time_limit = time_limit
        self.steps = 0
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        
        # Define action and observation space
        # Actions: click, type, press_enter, back, forward, scroll_down, scroll_up
        self.action_space = spaces.Discrete(7)
        
        # Observation space: screenshot of the page (as a normalized pixel array)
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.observation_height, self.observation_width, 3), 
            dtype=np.uint8
        )
        
        self.driver = None
        self.current_page_state = None
        self.last_reward = 0
        self.browser_initialized = False
    
    def _init_browser(self):
        """Initialize the Selenium browser."""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-notifications')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1024, 768)
        self.browser_initialized = True
        
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        # Close existing browser session if any
        if self.browser_initialized and self.driver:
            self.driver.quit()
        
        # Start a new browser session
        self._init_browser()
        self.driver.get(self.start_url)
        time.sleep(2)  # Wait for page to load
        
        self.steps = 0
        self.last_reward = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (int): The action to take:
                0: click a random clickable element
                1: type random text into a focused element
                2: press Enter
                3: navigate back
                4: navigate forward
                5: scroll down
                6: scroll up
        
        Returns:
            observation: Current screenshot of the page
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., time limit)
            info: Additional information
        """
        self.steps += 1
        terminated = False
        truncated = False
        reward = 0
        
        if self.steps >= self.time_limit:
            truncated = True
        
        try:
            if action == 0:  # Click a random clickable element
                reward = self._action_click()
            elif action == 1:  # Type random text
                reward = self._action_type()
            elif action == 2:  # Press Enter
                reward = self._action_press_enter()
            elif action == 3:  # Navigate back
                reward = self._action_back()
            elif action == 4:  # Navigate forward
                reward = self._action_forward()
            elif action == 5:  # Scroll down
                reward = self._action_scroll(scroll_down=True)
            elif action == 6:  # Scroll up
                reward = self._action_scroll(scroll_down=False)
            
        except Exception as e:
            logging.warning(f"Action error: {e}")
            reward = -1  # Penalty for failed actions
        
        # Wait for page to update
        time.sleep(0.5)
        
        # Get the new state, reward, etc.
        observation = self._get_observation()
        info = self._get_info()
        
        self.last_reward = reward
        
        return observation, reward, terminated, truncated, info
    
    def _action_click(self):
        """Click a random clickable element on the page."""
        try:
            # Find all clickable elements
            clickable_elements = self.driver.find_elements(By.XPATH, 
                "//a | //button | //input[@type='button'] | //input[@type='submit'] | //div[@role='button']")
            
            if not clickable_elements:
                return -0.1  # No clickable elements found
            
            # Select a random element
            import random
            element = random.choice(clickable_elements)
            
            # Try to scroll the element into view
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.5)
            
            # Click the element
            element.click()
            
            # Reward for successful navigation
            return 1.0
            
        except Exception as e:
            logging.warning(f"Click error: {e}")
            return -0.1
    
    def _action_type(self):
        """Type random text into a focused element."""
        try:
            # Find all input fields
            input_elements = self.driver.find_elements(By.XPATH, 
                "//input[@type='text'] | //textarea | //input[@type='search'] | //input[@type='email'] | //input[@type='password']")
            
            if not input_elements:
                return -0.1  # No input elements found
            
            # Select a random input element
            import random
            element = random.choice(input_elements)
            
            # Scroll the element into view
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.5)
            
            # Clear existing text and type new text
            element.clear()
            element.send_keys("sample search query")
            
            return 0.5  # Reward for successful typing
            
        except Exception as e:
            logging.warning(f"Type error: {e}")
            return -0.1
    
    def _action_press_enter(self):
        """Press Enter key on the focused element."""
        try:
            # Send Enter key to active element
            self.driver.switch_to.active_element.send_keys(Keys.ENTER)
            time.sleep(1)  # Wait for page to load
            return 0.5
            
        except Exception as e:
            logging.warning(f"Press Enter error: {e}")
            return -0.1
    
    def _action_back(self):
        """Navigate back in the browser history."""
        try:
            self.driver.back()
            time.sleep(1)  # Wait for page to load
            return 0.2
            
        except Exception as e:
            logging.warning(f"Back navigation error: {e}")
            return -0.1
    
    def _action_forward(self):
        """Navigate forward in the browser history."""
        try:
            self.driver.forward()
            time.sleep(1)  # Wait for page to load
            return 0.2
            
        except Exception as e:
            logging.warning(f"Forward navigation error: {e}")
            return -0.1
    
    def _action_scroll(self, scroll_down=True):
        """Scroll the page up or down."""
        try:
            direction = 300 if scroll_down else -300
            self.driver.execute_script(f"window.scrollBy(0, {direction});")
            return 0.1
            
        except Exception as e:
            logging.warning(f"Scroll error: {e}")
            return -0.1
    
    def _get_observation(self):
        """
        Get the current state of the environment as a screenshot.
        Preprocessed to match the observation space.
        """
        try:
            # Capture screenshot
            screenshot = self.driver.get_screenshot_as_png()
            
            # Convert to image
            image = Image.open(io.BytesIO(screenshot))
            
            # Resize to the dimensions expected by the observation space
            image = image.resize((self.observation_width, self.observation_height))
            
            # Convert to numpy array
            observation = np.array(image)
            
            return observation
            
        except Exception as e:
            logging.error(f"Error getting observation: {e}")
            # Return a blank image if there's an error
            return np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
    
    def _get_info(self):
        """Get additional information about the current state."""
        try:
            # Get current URL
            current_url = self.driver.current_url
            
            # Get page title
            title = self.driver.title
            
            # Get HTML source of the page
            page_source = self.driver.page_source
            
            # Parse with BeautifulSoup to extract more structured info
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Count links, buttons, and inputs
            links_count = len(soup.find_all('a'))
            buttons_count = len(soup.find_all('button'))
            inputs_count = len(soup.find_all('input'))
            
            # Create info dictionary
            info = {
                'url': current_url,
                'title': title,
                'links_count': links_count,
                'buttons_count': buttons_count,
                'inputs_count': inputs_count,
                'last_reward': self.last_reward,
                'steps': self.steps
            }
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting info: {e}")
            return {'error': str(e)}
    
    def render(self):
        """
        Render the environment.
        
        In 'human' mode, it displays the current browser window.
        In 'rgb_array' mode, it returns the current screenshot.
        """
        if self.render_mode == 'human':
            # The browser window is already visible in non-headless mode
            pass
        
        elif self.render_mode == 'rgb_array':
            return self._get_observation()
    
    def close(self):
        """Close the environment and browser."""
        if self.browser_initialized and self.driver:
            self.driver.quit()
            self.browser_initialized = False