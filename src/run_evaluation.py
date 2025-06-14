import os
from dotenv import load_dotenv
load_dotenv()

# Define global variable for Azure availability
AZURE_AVAILABLE = True

import time
import yaml
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from Levenshtein import distance as levenshtein_distance
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse

# Import all available models
from models.easyocr import EasyOCRModel
from models.tesseract import TesseractModel
from models.yolo_ocr import YOLOOCRModel
from models.azure_document_intelligence import AzureDocumentIntelligenceModel

# Attempt to import PaddleOCR module
try:
    from models.paddleocr import PaddleOCRModel
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"\nWarning: Could not import PaddleOCR module - {str(e)}")
    print("PaddleOCR will be excluded from evaluation.")
    PADDLEOCR_AVAILABLE = False

from preprocessing import (
    SharpeningPreprocessor,
    DenoisingPreprocessor,
    BinarizationPreprocessor
)
from utils.evaluation_utils import (
    create_evaluation_config,
    save_evaluation_results,
    load_all_results,
    generate_performance_report
)

def bbox_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def convert_bbox_to_x1y1x2y2(bbox, fmt='easyocr'):
    """Convert bounding box format to [x1, y1, x2, y2]."""
    if fmt == 'easyocr':
        # EasyOCR format is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] or similar quadrilateral
        # We need the min/max x and y
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif fmt == 'json':
        # JSON format is [x, y, width, height]
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    else:
        raise ValueError(f"Unknown bounding box format: {fmt}")

def load_test_data(config: Dict[str, Any]) -> tuple:
    """Load test data (including subfolders, loads all annotations)."""
    images = []
    ground_truth_annotations = [] # Store list of all annotations instead of list of texts
    
    data_dir = config['data']['test_dir']
    label_dir = config['data']['label_dir']
    test_data_limit = config['evaluation']['test_data_limit']

    # Match images and label files (explore subfolders)
    for root, _, files in os.walk(Path(data_dir) / 'images'): # Explore from images/ subfolder
        for file in files:
            if file.endswith('.jpg'):
                img_path = Path(root) / file
                
                # Get relative path from images/5350224/
                relative_img_path = img_path.relative_to(Path(data_dir) / 'images' / '5350224')
                
                # Construct corresponding label path
                json_path = Path(label_dir) / 'labels' / '5350224' / relative_img_path.parent / file.replace('.jpg', '.json')

                if json_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                        
                        # Store list of all annotations
                        ground_truth_annotations.append(label_data.get('annotations', []))
                        images.append(img)
                        
                        # Check if we've reached the limit
                        if test_data_limit > 0 and len(images) >= test_data_limit:
                            return images, ground_truth_annotations
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: No corresponding JSON file found for {img_path.name} at {json_path}")
    
    return images, ground_truth_annotations

def get_preprocessing_pipeline(steps: List[str]) -> List[Any]:
    """Creates a preprocessing pipeline."""
    pipeline = []
    for step in steps:
        if step == 'sharpening':
            pipeline.append(SharpeningPreprocessor())
        elif step == 'denoising':
            pipeline.append(DenoisingPreprocessor())
        elif step == 'binarization':
            pipeline.append(BinarizationPreprocessor())
    return pipeline

def evaluate_combination(
    model,
    images: List[np.ndarray],
    ground_truth: List[List[Dict[str, Any]]],
    preprocessing_steps: List[str]
) -> Dict[str, Any]:
    """Evaluates a specific model and preprocessing combination."""
    start_time = time.time()
    all_predictions = []
    
    # Create preprocessing pipeline
    pipeline = get_preprocessing_pipeline(preprocessing_steps)
    
    for img in images:
        # Apply preprocessing
        processed_img = img
        for preprocessor in pipeline:
            processed_img = preprocessor(processed_img)
        
        # Perform prediction
        pred = model(processed_img)
        all_predictions.append(pred)
    
    inference_time = time.time() - start_time
    
    # Calculate accuracy (bounding box based matching)
    total_items = 0
    matched_items = 0
    total_chars = 0
    matched_chars = 0
    
    # New metrics for text structure and layout
    total_lines = 0
    matched_lines = 0
    total_paragraphs = 0
    matched_paragraphs = 0
    layout_scores = []
    spatial_relationship_scores = []
    
    # Counters for additional metrics
    type_metrics = {}  # Accuracy by text type
    region_metrics = {}  # Accuracy by location (top/middle/bottom)
    length_metrics = {}  # Accuracy by text length (short/medium/long)
    size_metrics = {}  # Accuracy by bounding box size (small/medium/large)
    
    # Variables for full text comparison
    total_levenshtein_distance = 0
    total_gt_length = 0
    rouge = Rouge()
    total_rouge_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    total_rouge_count = 0
    
    # Variables for BLEU score
    total_bleu_score = 0
    total_bleu_count = 0
    smoothing = SmoothingFunction().method1
    
    # Variables for full text accuracy
    total_exact_matches = 0
    total_char_matches = 0
    total_word_matches = 0
    total_gt_chars = 0
    total_gt_words = 0
    total_normalized_matches = 0
    
    for pred_list, gt_annotations in zip(all_predictions, ground_truth):
        # Extract text and bounding boxes from Ground Truth annotations
        gt_texts = [anno.get('annotation.text', '') for anno in gt_annotations]
        gt_boxes = [convert_bbox_to_x1y1x2y2(anno.get('annotation.bbox', []), fmt='json') 
                   for anno in gt_annotations]
        gt_types = [anno.get('annotation.ttype', 'unknown') for anno in gt_annotations]
        
        # Group ground truth texts into lines and paragraphs
        gt_lines = []
        gt_paragraphs = []
        current_line = []
        current_paragraph = []
        
        for i, (text, box) in enumerate(zip(gt_texts, gt_boxes)):
            if i > 0:
                prev_box = gt_boxes[i-1]
                # Check if this text belongs to the same line
                if abs(box[1] - prev_box[1]) < 20:  # 20 pixels threshold for same line
                    current_line.append(text)
                else:
                    if current_line:
                        gt_lines.append(' '.join(current_line))
                        current_line = []
                    current_line.append(text)
                
                # Check if this text belongs to the same paragraph
                if abs(box[1] - prev_box[3]) < 50:  # 50 pixels threshold for same paragraph
                    current_paragraph.append(text)
                else:
                    if current_paragraph:
                        gt_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    current_paragraph.append(text)
            else:
                current_line.append(text)
                current_paragraph.append(text)
        
        if current_line:
            gt_lines.append(' '.join(current_line))
        if current_paragraph:
            gt_paragraphs.append(' '.join(current_paragraph))
        
        # Generate strings for full text comparison
        gt_full_text = ' '.join(gt_texts)
        pred_full_text = ' '.join([text for text, _ in pred_list])
        
        # Calculate full text accuracy metrics
        # 1. Exact Match
        if gt_full_text == pred_full_text:
            total_exact_matches += 1
        
        # 2. Character-level Accuracy
        total_gt_chars += len(gt_full_text)
        total_char_matches += sum(1 for c1, c2 in zip(gt_full_text, pred_full_text) if c1 == c2)
        
        # 3. Word-level Accuracy
        gt_words = gt_full_text.split()
        pred_words = pred_full_text.split()
        total_gt_words += len(gt_words)
        total_word_matches += sum(1 for w1, w2 in zip(gt_words, pred_words) if w1 == w2)
        
        # 4. Normalized Accuracy
        min_len = min(len(gt_full_text), len(pred_full_text))
        max_len = max(len(gt_full_text), len(pred_full_text))
        if max_len > 0:
            total_normalized_matches += sum(1 for c1, c2 in zip(gt_full_text[:min_len], pred_full_text[:min_len]) if c1 == c2) / max_len
        
        # Calculate line and paragraph accuracy
        pred_lines = [line.strip() for line in pred_full_text.split('\n') if line.strip()]
        total_lines += len(gt_lines)
        matched_lines += sum(1 for gt_line in gt_lines 
                           for pred_line in pred_lines 
                           if levenshtein_distance(gt_line, pred_line) < len(gt_line) * 0.3)
        
        # Calculate layout preservation score
        if len(gt_boxes) > 1:
            gt_spatial_relations = []
            pred_spatial_relations = []
            
            for i in range(len(gt_boxes)-1):
                for j in range(i+1, len(gt_boxes)):
                    # Calculate relative positions
                    gt_relation = (
                        gt_boxes[i][0] < gt_boxes[j][0],  # left/right
                        gt_boxes[i][1] < gt_boxes[j][1],  # top/bottom
                        abs(gt_boxes[i][1] - gt_boxes[j][1]) < 20  # same line
                    )
                    gt_spatial_relations.append(gt_relation)
            
            for i in range(len(pred_list)-1):
                for j in range(i+1, len(pred_list)):
                    pred_box_i = pred_list[i][1]
                    pred_box_j = pred_list[j][1]
                    pred_relation = (
                        pred_box_i[0] < pred_box_j[0],
                        pred_box_i[1] < pred_box_j[1],
                        abs(pred_box_i[1] - pred_box_j[1]) < 20
                    )
                    pred_spatial_relations.append(pred_relation)
            
            # Calculate spatial relationship score
            if gt_spatial_relations and pred_spatial_relations:
                spatial_score = sum(1 for gt, pred in zip(gt_spatial_relations, pred_spatial_relations)
                                  if gt == pred) / len(gt_spatial_relations)
                spatial_relationship_scores.append(spatial_score)
        
        # Calculate Levenshtein distance
        total_levenshtein_distance += levenshtein_distance(gt_full_text, pred_full_text)
        total_gt_length += len(gt_full_text)
        
        # Calculate ROUGE score
        try:
            if gt_full_text and pred_full_text:
                rouge_scores = rouge.get_scores(pred_full_text, gt_full_text)[0]
                for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                    total_rouge_scores[metric] += rouge_scores[metric]['f']
                total_rouge_count += 1
        except Exception as e:
            print(f"Warning: Error calculating ROUGE score - {str(e)}")
        
        # Calculate BLEU score
        try:
            if gt_full_text and pred_full_text:
                reference = [gt_full_text.split()]
                candidate = pred_full_text.split()
                weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
                bleu_scores = []
                
                for weight in weights:
                    score = sentence_bleu(reference, candidate, weights=weight, smoothing_function=smoothing)
                    bleu_scores.append(score)
                
                total_bleu_score += sum(bleu_scores) / len(bleu_scores)
                total_bleu_count += 1
        except Exception as e:
            print(f"Warning: Error calculating BLEU score - {str(e)}")
        
        total_items += len(gt_texts)
        
        # Match predictions with Ground Truth
        matched_gt_indices = set()
        for pred_text, pred_box in pred_list:
            best_iou = 0
            best_gt_idx = -1
            
            for i, (gt_text, gt_box) in enumerate(zip(gt_texts, gt_boxes)):
                if i in matched_gt_indices:
                    continue
                    
                iou = bbox_iou(pred_box, gt_box)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                matched_items += 1
                
                gt_text = gt_texts[best_gt_idx]
                total_chars += len(gt_text)
                matched_chars += sum(1 for c1, c2 in zip(pred_text, gt_text) if c1 == c2)
                
                gt_type = gt_types[best_gt_idx]
                if gt_type not in type_metrics:
                    type_metrics[gt_type] = {'total': 0, 'matched': 0}
                type_metrics[gt_type]['total'] += 1
                if pred_text == gt_text:
                    type_metrics[gt_type]['matched'] += 1
                
                y_center = (gt_box[1] + gt_box[3]) / 2
                region = 'top' if y_center < 0.33 else 'middle' if y_center < 0.66 else 'bottom'
                if region not in region_metrics:
                    region_metrics[region] = {'total': 0, 'matched': 0}
                region_metrics[region]['total'] += 1
                if pred_text == gt_text:
                    region_metrics[region]['matched'] += 1
                
                length = len(gt_text)
                length_key = 'short' if length <= 2 else 'medium' if length <= 5 else 'long'
                if length_key not in length_metrics:
                    length_metrics[length_key] = {'total': 0, 'matched': 0}
                length_metrics[length_key]['total'] += 1
                if pred_text == gt_text:
                    length_metrics[length_key]['matched'] += 1
                
                box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                size_key = 'small' if box_area < 1000 else 'medium' if box_area < 5000 else 'large'
                if size_key not in size_metrics:
                    size_metrics[size_key] = {'total': 0, 'matched': 0}
                size_metrics[size_key]['total'] += 1
                if pred_text == gt_text:
                    size_metrics[size_key]['matched'] += 1
    
    # Calculate final accuracies
    item_accuracy = float(matched_items) / total_items if total_items > 0 else 0.0
    char_accuracy = float(matched_chars) / total_chars if total_chars > 0 else 0.0
    line_accuracy = float(matched_lines) / total_lines if total_lines > 0 else 0.0
    spatial_relationship_score = float(sum(spatial_relationship_scores)) / len(spatial_relationship_scores) if spatial_relationship_scores else 0.0
    
    # Calculate full text accuracy metrics
    full_text_accuracy = {
        'exact_match': float(total_exact_matches) / len(all_predictions) if all_predictions else 0.0,
        'char_accuracy': float(total_char_matches) / total_gt_chars if total_gt_chars > 0 else 0.0,
        'word_accuracy': float(total_word_matches) / total_gt_words if total_gt_words > 0 else 0.0,
        'normalized_accuracy': float(total_normalized_matches) / len(all_predictions) if all_predictions else 0.0
    }
    
    # Calculate additional metrics
    type_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                      for k, v in type_metrics.items()}
    region_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                        for k, v in region_metrics.items()}
    length_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                        for k, v in length_metrics.items()}
    size_accuracies = {k: float(v['matched']) / v['total'] if v['total'] > 0 else 0.0 
                      for k, v in size_metrics.items()}
    
    # Calculate full text comparison metrics
    normalized_levenshtein = 1.0 - (float(total_levenshtein_distance) / total_gt_length) if total_gt_length > 0 else 0.0
    rouge_scores = {k: float(v) / total_rouge_count if total_rouge_count > 0 else 0.0 
                   for k, v in total_rouge_scores.items()}
    bleu_score = float(total_bleu_score) / total_bleu_count if total_bleu_count > 0 else 0.0

    return {
        'metrics': {
            'item_accuracy': item_accuracy,
            'char_accuracy': char_accuracy,
            'line_accuracy': line_accuracy,
            'spatial_relationship_score': spatial_relationship_score,
            'inference_time': inference_time,
            'type_accuracies': type_accuracies,
            'region_accuracies': region_accuracies,
            'length_accuracies': length_accuracies,
            'size_accuracies': size_accuracies,
            'text_similarity': {
                'normalized_levenshtein': normalized_levenshtein,
                'rouge_scores': rouge_scores,
                'bleu_score': bleu_score
            },
            'full_text_accuracy': full_text_accuracy
        },
        'predictions': all_predictions
    }

def main():
    # Add argparse for model selection
    parser = argparse.ArgumentParser(description="OCR Evaluation")
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to evaluate (e.g., yolo_ocr,azure_read)')
    args = parser.parse_args()

    config = create_evaluation_config()
    
    # Initialize all available models
    all_models = {
        'tesseract': TesseractModel(),
        'easyocr': EasyOCRModel(),
        'yolo_ocr': YOLOOCRModel()
    }
    if PADDLEOCR_AVAILABLE:
        all_models['paddleocr'] = PaddleOCRModel()
    if AZURE_AVAILABLE:
        try:
            all_models['azure_read'] = AzureDocumentIntelligenceModel(mode='read')
            all_models['azure_layout'] = AzureDocumentIntelligenceModel(mode='layout')
            all_models['azure_prebuilt_read'] = AzureDocumentIntelligenceModel(mode='prebuilt_read')
        except Exception as e:
            print(f"\nWarning: Azure Document Intelligence not configured - {str(e)}")
            print("Azure Document Intelligence will be excluded from evaluation.")
    
    # Filter models if --models is specified
    if args.models:
        selected = [m.strip() for m in args.models.split(',')]
        models = {k: v for k, v in all_models.items() if k in selected}
        if not models:
            print(f"No valid models selected from: {selected}")
            return
    else:
        models = all_models
    
    # Load test data
    print("\nLoading test data...")
    images, ground_truth = load_test_data(config)
    print(f"Loaded {len(images)} test images")
    
    # Define preprocessing combinations
    preprocessing_combinations = [
        [],  # No preprocessing
        ['sharpening'],
        ['denoising'],
        ['binarization'],
        ['sharpening', 'denoising'],
        ['sharpening', 'binarization'],
        ['denoising', 'binarization'],
        ['sharpening', 'denoising', 'binarization']
    ]
    
    # Run evaluation for each model and preprocessing combination
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        for preproc_steps in preprocessing_combinations:
            preproc_name = '+'.join(preproc_steps) if preproc_steps else 'none'
            print(f"  With preprocessing: {preproc_name}")
            result = evaluate_combination(model, images, ground_truth, preproc_steps)
            # Save each result as a separate file with correct config
            save_evaluation_results(
                result,
                {
                    'model_name': model_name,
                    'preprocessing_steps': preproc_steps
                }
            )
            print(f"Evaluation results saved for {model_name} with preprocessing: {preproc_name}")
    
    # Generate and print performance report from all results in results dir
    print("\nGenerating performance report...")
    all_results = load_all_results()
    generate_performance_report(all_results)
    print("\nPerformance report saved as CSV in results/ directory.")

if __name__ == "__main__":
    main() 