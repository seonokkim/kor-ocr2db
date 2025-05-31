import os
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any, List
import time
import json

# Import training functionality from other models
# from models.tesseract import TesseractModel # example
# from models.paddleocr import PaddleOCRModel # example

def load_training_data(train_dir: str) -> tuple:
    """Load training data (including subfolders)."""
    images = []
    labels = []
    
    # Training data loading logic (explore subfolders)
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                # Example: Load .json file with same name as image file (maintain relative path)
                label_path = (Path(train_dir) / Path(root).relative_to(train_dir) / file).with_suffix('.json').parent.parent / Path('labels') / Path(root).relative_to(Path(train_dir) / 'images') / file.replace('.jpg', '.json')
                
                # Debug print for label file path
                # print(f"Checking label path: {label_path}")

                if label_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                         with open(label_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            # Extract and combine 'annotation.text' values from JSON data
                            text = " ".join([anno.get('annotation.text', '') for anno in label_data.get('annotations', [])])
                            labels.append(text)
                            images.append(img)
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: No corresponding JSON file found for {img_path} at {label_path}")

    return images, labels

def train_model(model_name: str, train_data: tuple, config: Dict[str, Any], save_path: str):
    """Train and save the model with given data."""
    print(f"\nStarting model training: {model_name}")
    
    images, labels = train_data
    
    if not images:
        print("No training data available. Skipping training.")
        return

    # Actual training logic implementation
    # EasyOCR has limited API for user training.
    # Here we present a conceptual training structure, but actual training should follow
    # the library's guide or use other models that support user training (e.g., Tesseract, PaddleOCR).

    if model_name == 'tesseract':
        print("Tesseract model training requires separate implementation (using tesseract train tool).")
        # Implement logic to train and save model using Tesseract training tools
        pass
    elif model_name == 'paddleocr':
        print("PaddleOCR model training requires separate implementation (using PaddleOCR training framework).")
        # Implement logic to train and save model using PaddleOCR training tools
        pass
    # Add other model training logic
    else:
        print(f"{model_name} model does not support user training or training code is not implemented.")
        print(f"Not saving trained model to {save_path}.")
        return
    
    print(f"{model_name} model training completed. Saved trained model to {save_path}.")

def main():
    # Load configuration
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load training data
    train_images, train_labels = load_training_data(config['data']['train_dir'])
    print(f"Loaded training images: {len(train_images)}")

    # Path for saving trained models
    trained_model_dir = "trained_models"
    os.makedirs(trained_model_dir, exist_ok=True)
    
    # Train models defined in config that support training
    learnable_models = [m for m in config['models']['available'] if m in ['tesseract', 'paddleocr']] # List of models supporting user training

    for model_name in learnable_models:
        model_save_path = os.path.join(trained_model_dir, f'{model_name}_korean')
        train_model(model_name, (train_images, train_labels), config, model_save_path)

if __name__ == "__main__":
    main() 