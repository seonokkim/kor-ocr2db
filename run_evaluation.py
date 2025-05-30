import yaml
import os
from utils import load_images_from_dir, load_labels_from_dir, calculate_metrics
import easyocr
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json

def preprocess_image(image: np.ndarray, steps: List[str]) -> np.ndarray:
    """Apply image preprocessing steps"""
    processed = image.copy()
    
    for step in steps:
        if step == 'sharpening':
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
        elif step == 'contrast':
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        elif step == 'denoising':
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    return processed

def load_images_from_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    """Load images from directory and subdirectories with their filenames"""
    images = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Use the full path relative to data directory as filename
                    rel_path = os.path.relpath(img_path, directory)
                    images.append((rel_path, img))
                    print(f"[DEBUG run_evaluation.py] Loaded image key: '{rel_path}'")
    return images

def load_labels_from_dir(directory: str) -> Dict[str, str]:
    """Load ground truth labels from text files in subdirectories"""
    labels = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.txt'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Use the full path relative to labels directory as key
                    rel_path = os.path.relpath(filepath, directory)
                    # Remove .txt extension for matching
                    key = os.path.splitext(rel_path)[0]
                    labels[key] = f.read().strip()
    return labels

def evaluate_ocr(config: Dict):
    """Run OCR evaluation with different models and preprocessing steps"""
    results = []
    
    # Load data
    images = load_images_from_dir(os.path.join(config['data_directory'], 'images'))
    labels = load_labels_from_dir(os.path.join(config['data_directory'], 'labels'))
    
    # Print loaded data statistics
    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(labels)} labels")
    
    # Initialize OCR reader
    reader = easyocr.Reader(['ko'], gpu=config.get('use_gpu', False), verbose=False)
    
    # Create output directory
    os.makedirs(config['output_directory'], exist_ok=True)
    
    # Process each image
    for filename, image in images:
        # Use the filename without extension to match labels
        filename_without_ext = os.path.splitext(filename)[0]
        if filename_without_ext not in labels:
            print(f"Warning: No label found for image: {filename}")
            continue
            
        gt_text = labels[filename_without_ext]
        
        # Evaluate raw image
        for model_name in config['models_to_evaluate']:
            if model_name == 'easyocr':
                result = reader.readtext(image)
                pred_text = ' '.join([text[1] for text in result])
            elif model_name == 'tesseract':
                pred_text = pytesseract.image_to_string(image, lang='kor')
            
            metrics = calculate_metrics(gt_text, pred_text)
            
            # Process image and evaluate again
            processed_image = preprocess_image(image, config['image_processing_steps'])
            
            if model_name == 'easyocr':
                result_processed = reader.readtext(processed_image)
                pred_text_processed = ' '.join([text[1] for text in result_processed])
            elif model_name == 'tesseract':
                pred_text_processed = pytesseract.image_to_string(processed_image, lang='kor')
            
            metrics_processed = calculate_metrics(gt_text, pred_text_processed)
            
            results.append({
                'filename': filename,
                'model': model_name,
                'ground_truth': gt_text,
                'prediction_raw': pred_text,
                'prediction_processed': pred_text_processed,
                'metrics_raw': metrics,
                'metrics_processed': metrics_processed
            })
    
    # Save results
    with open(os.path.join(config['output_directory'], 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Calculate and print average metrics
    avg_metrics = {model: {'raw': {}, 'processed': {}} for model in config['models_to_evaluate']}
    
    for result in results:
        model = result['model']
        for metric in config['metrics']:
            if metric not in avg_metrics[model]['raw']:
                avg_metrics[model]['raw'][metric] = []
                avg_metrics[model]['processed'][metric] = []
            
            avg_metrics[model]['raw'][metric].append(result['metrics_raw'][metric])
            avg_metrics[model]['processed'][metric].append(result['metrics_processed'][metric])
    
    print("\nAverage OCR Performance Metrics:")
    for model in config['models_to_evaluate']:
        print(f"\nModel: {model}")
        print("Raw Images:")
        for metric in config['metrics']:
            avg = sum(avg_metrics[model]['raw'][metric]) / len(avg_metrics[model]['raw'][metric])
            print(f"  {metric}: {avg:.4f}")
        print("Processed Images:")
        for metric in config['metrics']:
            avg = sum(avg_metrics[model]['processed'][metric]) / len(avg_metrics[model]['processed'][metric])
            print(f"  {metric}: {avg:.4f}")

if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    evaluate_ocr(config)
