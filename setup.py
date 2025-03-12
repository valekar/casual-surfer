import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("All requirements installed successfully!")

def setup_browser_drivers():
    """Setup browser drivers for Selenium"""
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium import webdriver
        
        print("Setting up Chrome WebDriver...")
        # This will download and install the appropriate ChromeDriver
        driver_path = ChromeDriverManager().install()
        print(f"Chrome WebDriver installed at: {driver_path}")
        
        # Test if the driver works
        print("Testing WebDriver...")
        driver = webdriver.Chrome()
        driver.get("https://www.google.com")
        driver.quit()
        print("WebDriver test successful!")
        
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        print("Please make sure Chrome browser is installed on your system.")

if __name__ == "__main__":
    install_requirements()
    setup_browser_drivers()
    
    # Create project directories
    directories = ["data", "models", "logs", "configs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nSetup completed successfully! You're ready to start building your RL browser agent.")