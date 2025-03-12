import os
import json
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/data_processing.log"),
        logging.StreamHandler()
    ]
)

class BrowserDataProcessor:
    """
    Processes collected browser interaction data to create features
    suitable for training reinforcement learning models.
    """
    
    def __init__(self, data_dir="data", output_dir="data/processed"):
        """
        Initialize the data processor.
        
        Args:
            data_dir (str): Directory containing the collected data
            output_dir (str): Directory to save processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize image preprocessing model (feature extractor)
        self.feature_extractor = None
        
        # Containers for processed data
        self.data_df = None
        self.features = None
    
    def load_data(self, pattern="browser_data_session_*.json"):
        """
        Load all data files matching the given pattern.
        
        Args:
            pattern (str): Glob pattern to match data files
        
        Returns:
            int: Number of interactions loaded
        """
        logging.info(f"Loading data from {self.data_dir}...")
        
        # Find all matching files
        file_pattern = os.path.join(self.data_dir, pattern)
        data_files = glob.glob(file_pattern)
        
        if not data_files:
            logging.warning(f"No data files found matching {file_pattern}")
            return 0
        
        logging.info(f"Found {len(data_files)} data files")
        
        # Load and merge all data
        all_data = []
        
        for file_path in tqdm(data_files, desc="Loading data files"):
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    all_data.extend(file_data)
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
        
        logging.info(f"Loaded {len(all_data)} total interactions")
        
        # Convert to pandas DataFrame for easier processing
        rows = []
        
        for interaction in all_data:
            try:
                # Extract key information
                row = {
                    "session_id": interaction.get("session_id", -1),
                    "interaction_id": interaction.get("interaction_id", -1),
                    "timestamp": interaction.get("timestamp", ""),
                    "url": interaction.get("state", {}).get("url", ""),
                    "title": interaction.get("state", {}).get("title", ""),
                    "screenshot_path": interaction.get("state", {}).get("screenshot_path", ""),
                    "links_count": interaction.get("state", {}).get("links_count", 0),
                    "buttons_count": interaction.get("state", {}).get("buttons_count", 0),
                    "inputs_count": interaction.get("state", {}).get("inputs_count", 0),
                    "selects_count": interaction.get("state", {}).get("selects_count", 0),
                    "action_type": interaction.get("action", {}).get("type", ""),
                    "action_success": interaction.get("action", {}).get("success", False),
                }
                
                # Add to rows
                rows.append(row)
                
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
        
        # Create DataFrame
        self.data_df = pd.DataFrame(rows)
        logging.info(f"Created DataFrame with {len(self.data_df)} rows")
        
        return len(self.data_df)
    
    def preprocess_data(self):
        """
        Preprocess the loaded data:
        - Clean missing values
        - Filter out failed interactions
        - Process URL and title features
        - Normalize numeric features
        """
        if self.data_df is None or len(self.data_df) == 0:
            logging.error("No data loaded. Call load_data() first.")
            return False
        
        logging.info("Preprocessing data...")
        
        # Filter out rows with missing screenshots
        original_count = len(self.data_df)
        self.data_df = self.data_df[self.data_df["screenshot_path"].apply(
            lambda x: isinstance(x, str) and os.path.exists(x)
        )]
        logging.info(f"Filtered out {original_count - len(self.data_df)} rows with missing screenshots")
        
        # Filter out failed actions
        original_count = len(self.data_df)
        self.data_df = self.data_df[self.data_df["action_success"] == True]
        logging.info(f"Filtered out {original_count - len(self.data_df)} failed actions")
        
        # Process URLs - extract domain and path length
        self.data_df["domain"] = self.data_df["url"].apply(self._extract_domain)
        self.data_df["path_length"] = self.data_df["url"].apply(self._extract_path_length)
        
        # Create action type one-hot encoding
        action_dummies = pd.get_dummies(self.data_df["action_type"], prefix="action")
        self.data_df = pd.concat([self.data_df, action_dummies], axis=1)
        
        # Normalize numeric features
        numeric_features = ["links_count", "buttons_count", "inputs_count", "selects_count", "path_length"]
        scaler = StandardScaler()
        self.data_df[numeric_features] = scaler.fit_transform(self.data_df[numeric_features])
        
        logging.info(f"Preprocessing complete. Final dataset shape: {self.data_df.shape}")
        
        # Save preprocessed data
        processed_path = os.path.join(self.output_dir, "preprocessed_data.csv")
        self.data_df.to_csv(processed_path, index=False)
        logging.info(f"Saved preprocessed data to {processed_path}")
        
        return True
    
    def extract_visual_features(self, img_size=(224, 224), batch_size=32):
        """
        Extract visual features from screenshots using a pre-trained CNN.
        
        Args:
            img_size (tuple): Size to resize images to
            batch_size (int): Batch size for feature extraction
        
        Returns:
            bool: Success or failure
        """
        if self.data_df is None or len(self.data_df) == 0:
            logging.error("No data loaded. Call load_data() first.")
            return False
        
        logging.info(f"Extracting visual features from {len(self.data_df)} screenshots...")
        
        # Initialize feature extractor if needed
        if self.feature_extractor is None:
            logging.info("Initializing MobileNetV2 feature extractor...")
            base_model = MobileNetV2(
                input_shape=(img_size[0], img_size[1], 3),
                include_top=False,
                pooling='avg'
            )
            self.feature_extractor = base_model
        
        # Process images in batches
        all_features = []
        all_ids = []
        
        for i in tqdm(range(0, len(self.data_df), batch_size), desc="Extracting features"):
            batch = self.data_df.iloc[i:i+batch_size]
            
            # Load and preprocess images
            batch_images = []
            batch_interaction_ids = []
            
            for _, row in batch.iterrows():
                try:
                    # Load and preprocess image
                    img_path = row["screenshot_path"]
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    img_array = img_to_array(img)
                    img_array = preprocess_input(img_array)
                    
                    batch_images.append(img_array)
                    batch_interaction_ids.append((row["session_id"], row["interaction_id"]))
                    
                except Exception as e:
                    logging.warning(f"Error processing image {row['screenshot_path']}: {e}")
            
            if not batch_images:
                continue
            
            # Convert to array and extract features
            batch_images = np.array(batch_images)
            batch_features = self.feature_extractor.predict(batch_images, verbose=0)
            
            all_features.append(batch_features)
            all_ids.extend(batch_interaction_ids)
        
        if not all_features:
            logging.error("No features extracted.")
            return False
        
        # Combine all batches
        all_features = np.vstack(all_features)
        
        logging.info(f"Extracted features with shape: {all_features.shape}")
        
        # Save features
        features_path = os.path.join(self.output_dir, "visual_features.npz")
        np.savez(
            features_path,
            features=all_features,
            ids=all_ids
        )
        logging.info(f"Saved visual features to {features_path}")
        
        # Store in instance
        self.features = all_features
        
        return True
    
    def create_training_dataset(self):
        """
        Create the final training dataset by combining tabular data with visual features.
        
        Returns:
            tuple: (X, y) - Features and target values
        """
        if self.data_df is None or len(self.data_df) == 0:
            logging.error("No data loaded. Call load_data() first.")
            return None, None
        
        # Load visual features if not already in memory
        if self.features is None:
            features_path = os.path.join(self.output_dir, "visual_features.npz")
            if not os.path.exists(features_path):
                logging.error(f"Visual features not found at {features_path}. Run extract_visual_features() first.")
                return None, None
            
            data = np.load(features_path)
            self.features = data["features"]
            feature_ids = data["ids"]
            
            logging.info(f"Loaded visual features with shape: {self.features.shape}")
        else:
            feature_ids = [(row["session_id"], row["interaction_id"]) for _, row in self.data_df.iterrows()]
        
        # Create the target values (rewards)
        # These are simplified for this example - in a real-world scenario,
        # you would derive these from actual user behavior or outcomes
        
        # For our simple example:
        # - Successful navigation (click followed by URL change): +1
        # - Successful form interaction (type + enter): +0.5
        # - Scroll actions: +0.1
        # - Others: +0.2
        
        def calculate_reward(row):
            if row["action_type"] == "click" and row.get("action_details", {}).get("result_url", "") != row["url"]:
                return 1.0
            elif row["action_type"] == "type" and row.get("action_type_next") == "press_enter":
                return 0.5
            elif row["action_type"].startswith("scroll"):
                return 0.1
            else:
                return 0.2
        
        self.data_df["reward"] = self.data_df.apply(calculate_reward, axis=1)
        
        # Combine tabular features with visual features
        # Note: This assumes the order of self.data_df matches the order of self.features
        
        # Select the columns to use as tabular features
        tabular_cols = [
            "links_count", "buttons_count", "inputs_count", "selects_count", "path_length"
        ] + [col for col in self.data_df.columns if col.startswith("action_")]
        
        X_tabular = self.data_df[tabular_cols].values
        
        # Concatenate with visual features
        X = np.hstack([self.features, X_tabular])
        
        # Target values
        y = self.data_df["reward"].values
        
        logging.info(f"Created training dataset: X shape {X.shape}, y shape {y.shape}")
        
        # Save the final dataset
        dataset_path = os.path.join(self.output_dir, "training_dataset.npz")
        np.savez(
            dataset_path,
            X=X,
            y=y,
            feature_columns=tabular_cols
        )
        logging.info(f"Saved training dataset to {dataset_path}")
        
        return X, y
    
    def _extract_domain(self, url):
        """Extract the domain from a URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    def _extract_path_length(self, url):
        """Extract the path length from a URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return len(parsed.path.split("/"))
        except:
            return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process browser interaction data for RL training")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing collected data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
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