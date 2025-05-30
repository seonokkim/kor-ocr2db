import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import pytesseract

def calculate_metrics(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    Calculate OCR performance metrics
    Returns:
        Dict containing:
        - accuracy: Character-level accuracy
        - wer: Word Error Rate
        - cer: Character Error Rate
        - recall: Recall score
        - precision: Precision score
    """
    # Calculate character-level accuracy
    correct_chars = sum(1 for g, p in zip(gt_text, pred_text) if g == p)
    total_chars = max(len(gt_text), len(pred_text))
    accuracy = correct_chars / total_chars if total_chars > 0 else 0

    # Calculate Word Error Rate (WER)
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    wer = 1 - (len(set(gt_words) & set(pred_words)) / len(gt_words)) if gt_words else 1

    # Calculate Character Error Rate (CER)
    cer = 1 - (correct_chars / len(gt_text)) if gt_text else 1

    # Calculate recall and precision
    tp = len(set(gt_words) & set(pred_words))
    fp = len(pred_words) - tp
    fn = len(gt_words) - tp
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'wer': wer,
        'cer': cer,
        'recall': recall,
        'precision': precision
    }

def load_images_from_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    """Load images from directory with their filenames"""
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    return images

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Basic image preprocessing for OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    return processed

def load_labels_from_dir(directory: str) -> Dict[str, str]:
    """Load ground truth labels from text files, searching subdirectories."""
    labels = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.txt'):
                filepath = os.path.join(root, filename)
                # Create a key that matches the relative path of the image
                # (relative to the base 'labels' directory specified in config)
                rel_path = os.path.relpath(filepath, directory)
                # Remove the .txt extension for the key
                label_key = os.path.splitext(rel_path)[0]
                with open(filepath, 'r', encoding='utf-8') as f:
                    labels[label_key] = f.read().strip()
                    print(f"[DEBUG utils.py] Loaded label key: '{label_key}'")
    return labels
