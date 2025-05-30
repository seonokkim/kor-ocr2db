import os
import json
import pandas as pd
from utils import calculate_metrics, load_images_from_dir, load_labels_from_dir
from paddleocr import PaddleOCR
from transformers import pipeline
import pytesseract
from typing import Dict, List

def evaluate_ocr_models(data_dir: str):
    """
    Evaluate OCR performance using multiple models
    """
    # Load images and labels
    images = load_images_from_dir(os.path.join(data_dir, 'images'))
    labels = load_labels_from_dir(os.path.join(data_dir, 'labels'))
    
    # Initialize models
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ko')
    huggingface_ocr = pipeline('ocr', model='microsoft/trocr-large-printed')
    
    # Store results
    results = []
    
    for filename, image in images:
        if filename not in labels:
            continue
            
        gt_text = labels[filename]
        
        # 1. Evaluate raw image with different models
        # Paddle OCR
        paddle_result = paddle_ocr.ocr(image, cls=True)
        paddle_text = ' '.join([line[1][0] for line in paddle_result[0]])
        paddle_metrics = calculate_metrics(gt_text, paddle_text)
        
        # HuggingFace OCR
        hf_result = huggingface_ocr(image)
        hf_text = hf_result[0]['text'] if hf_result else ''
        hf_metrics = calculate_metrics(gt_text, hf_text)
        
        # Tesseract OCR
        tess_text = pytesseract.image_to_string(image, lang='kor')
        tess_metrics = calculate_metrics(gt_text, tess_text)
        
        # 2. Evaluate processed image
        processed_image = preprocess_image(image)
        
        # Paddle OCR on processed
        paddle_result_processed = paddle_ocr.ocr(processed_image, cls=True)
        paddle_text_processed = ' '.join([line[1][0] for line in paddle_result_processed[0]])
        paddle_metrics_processed = calculate_metrics(gt_text, paddle_text_processed)
        
        # Store results
        results.append({
            'filename': filename,
            'gt_text': gt_text,
            'paddle_text': paddle_text,
            'hf_text': hf_text,
            'tesseract_text': tess_text,
            'paddle_text_processed': paddle_text_processed,
            'paddle_metrics': paddle_metrics,
            'hf_metrics': hf_metrics,
            'tesseract_metrics': tess_metrics,
            'paddle_metrics_processed': paddle_metrics_processed
        })
    
    # Calculate average metrics
    avg_metrics = {
        'model': 'Average',
        'paddle': {},
        'hf': {},
        'tesseract': {},
        'paddle_processed': {}
    }
    
    for result in results:
        for model in ['paddle', 'hf', 'tesseract', 'paddle_processed']:
            metrics = result[f'{model}_metrics']
            for metric, value in metrics.items():
                if metric not in avg_metrics[model]:
                    avg_metrics[model][metric] = []
                avg_metrics[model][metric].append(value)
    
    for model in avg_metrics:
        if model == 'model':
            continue
        for metric in avg_metrics[model]:
            avg_metrics[model][metric] = sum(avg_metrics[model][metric]) / len(avg_metrics[model][metric])
    
    return results, avg_metrics

def main():
    results, avg_metrics = evaluate_ocr_models('data')
    
    # Save detailed results
    with open('ocr_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Print average metrics
    print("\nAverage OCR Performance Metrics:")
    for model, metrics in avg_metrics.items():
        if model == 'model':
            continue
        print(f"\n{model.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
