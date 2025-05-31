import os
import time
import yaml
from typing import List, Dict, Any
import cv2
import numpy as np
from pathlib import Path

from models import EasyOCRModel
from preprocessing import SharpeningPreprocessor
from utils.evaluation_utils import (
    create_evaluation_config,
    save_evaluation_results,
    load_all_results,
    generate_performance_report
)

def load_config(config_path: str = "configs/default_config.yaml") -> Dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_images(data_dir: str) -> List[np.ndarray]:
    """Load test images."""
    images = []
    for img_path in Path(data_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    return images

def evaluate_model(
    model,
    images: List[np.ndarray],
    ground_truth: List[str],
    preprocessing_steps: List[str] = None
) -> Dict[str, Any]:
    """Evaluate the model."""
    start_time = time.time()
    predictions = []
    
    for img in images:
        # Apply preprocessing
        processed_img = img
        if preprocessing_steps:
            for step in preprocessing_steps:
                if step == 'sharpening':
                    preprocessor = SharpeningPreprocessor()
                    processed_img = preprocessor(processed_img)
                # Add other preprocessing steps as needed
        
        # Perform prediction
        pred = model(processed_img)
        predictions.extend(pred)
    
    inference_time = time.time() - start_time
    
    # Calculate accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0
    
    # Calculate character-level accuracy
    total_chars = sum(len(g) for g in ground_truth)
    correct_chars = sum(sum(1 for c1, c2 in zip(p, g) if c1 == c2) 
                       for p, g in zip(predictions, ground_truth))
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    return {
        'metrics': {
            'accuracy': accuracy,
            'char_accuracy': char_accuracy,
            'inference_time': inference_time
        },
        'predictions': predictions
    }

def main():
    # Load configuration
    config = load_config()
    
    # Load test data
    test_images = load_test_images(config['data']['test_dir'])
    # TODO: Implement ground_truth data loading
    
    # Initialize models
    models = {
        'easyocr': EasyOCRModel(),
        # Add other models as needed
    }
    
    # Preprocessing step combinations
    preprocessing_combinations = [
        [],  # No preprocessing
        ['sharpening'],
        # Add other preprocessing combinations as needed
    ]
    
    # Perform evaluation for all combinations
    for model_name, model in models.items():
        for preprocess_steps in preprocessing_combinations:
            # Create evaluation configuration
            eval_config = create_evaluation_config(
                model_name=model_name,
                preprocessing_steps=preprocess_steps,
                use_gpu=config['hardware']['use_gpu']
            )
            
            # Perform evaluation
            results = evaluate_model(
                model=model,
                images=test_images,
                ground_truth=[],  # TODO: Add ground_truth data
                preprocessing_steps=preprocess_steps
            )
            
            # Save results
            save_evaluation_results(results, eval_config)
    
    # Analyze all results and generate report
    all_results = load_all_results()
    report = generate_performance_report(all_results)
    print("Evaluation completed! Results can be found in the results directory.")

if __name__ == "__main__":
    main() 