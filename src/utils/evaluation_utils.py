import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np

# Helper function to convert numpy types to standard Python types
def convert_numpy_types(obj):
    """Recursively convert numpy types to standard Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

def create_evaluation_config() -> Dict[str, Any]:
    """Create evaluation configuration with default settings."""
    return {
        'data': {
            'test_dir': 'data/test',
            'label_dir': 'data/test',
            'test_data_limit': 1  # Limit to 1 test sample
        },
        'evaluation': {
            'iou_threshold': 0.5,
            'test_data_limit': 1,
            'preprocessing_steps': []  # Default empty preprocessing steps
        },
        'hardware': {
            'use_gpu': True
        }
    }

def get_next_result_number(model_name: str, preprocess_info: str) -> int:
    """Generates the next result file number for a given model and preprocessing combo.
        
    Args:
        model_name (str): Name of the model.
        preprocess_info (str): Preprocessing information string.
        
    Returns:
        int: The next sequential file number.
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find files from today with the same model and preprocessing info
    today = time.strftime('%Y%m%d')
    pattern = f"{today}_{model_name}_{preprocess_info}_*.json"
    existing_files = list(results_dir.glob(pattern))
    
    if not existing_files:
        return 1
    
    # Find the highest number
    numbers = [int(f.stem.split('_')[-1]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def get_next_report_number() -> int:
    """Generates the next performance report file number.
        
    Returns:
        int: The next sequential file number.
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find report files from today
    today = time.strftime('%Y%m%d')
    pattern = f"{today}_performance_report_*.csv"
    existing_files = list(results_dir.glob(pattern))
    
    if not existing_files:
        return 1
    
    # Find the highest number
    numbers = [int(f.stem.split('_')[-1]) for f in existing_files]
    return max(numbers) + 1 if numbers else 1

def save_evaluation_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Saves evaluation results to a JSON file.
        
    Args:
        results (Dict[str, Any]): Dictionary containing evaluation results.
        config (Dict[str, Any]): Evaluation configuration dictionary.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename: timestamp_modelname_preprocessingcombo_sequence.json
    model_name = config.get('model_name', 'unknown')
    preprocess_tag = "_".join(config.get('preprocessing_steps', [])) if config.get('preprocessing_steps') else "no_preprocess"
    today = time.strftime('%Y%m%d')
    next_num = get_next_result_number(model_name, preprocess_tag)
    filename = f"{today}_{model_name}_{preprocess_tag}_{next_num}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert all numpy types in the results dictionary
    serializable_results = convert_numpy_types({
        'config': config,
        'metrics': results.get('metrics', {}),
        'predictions': results.get('predictions', [])
    })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation results saved to {filepath}")

def load_all_results() -> Dict[str, Any]:
    """Loads all evaluation results from the results directory.
        
    Returns:
        Dict[str, Any]: Dictionary containing all loaded results.
    """
    all_results = {}
    results_dir = "results"
    if not os.path.exists(results_dir):
        return all_results
    
    # Search for all JSON files (currently not considering subfolders)
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
                # Extract config information from filename (e.g., 20231027_base_easyocr_sharpening_1.json)
                filename = os.path.basename(filepath)
                parts = filename.replace('.json', '').split('_')
                
                # Using filename as key for easy management and to prevent duplicates
                config_key = filename
                
                # Separate metrics information from config information
                metrics = result_data.get('metrics', {})
                
                # Use config information stored within the file, fall back to filename parsing if needed
                eval_config_from_file = {
                     'config_name': result_data.get('config', {}).get('config_name', filename),
                     'model_name': result_data.get('config', {}).get('model_name', parts[1] if len(parts) > 1 else 'unknown'),
                     'preprocessing_steps': result_data.get('config', {}).get('preprocessing_steps', parts[2:-1] if len(parts) > 3 else []), # Estimate from filename
                     'use_gpu': result_data.get('config', {}).get('use_gpu', False), # Default value
                     'timestamp': result_data.get('config', {}).get('timestamp', parts[0] if len(parts) > 0 else '')
                }
                
                all_results[config_key] = {
                    'config': eval_config_from_file,
                    'metrics': metrics
                }
                
        except Exception as e:
            print(f"Warning: Failed to load results from {filepath}: {e}")
            
    return all_results

def plot_performance_comparison(df: pd.DataFrame, metric: str = 'item_accuracy'):
    """Generate performance comparison graph."""
    plt.figure(figsize=(12, 6))
    
    # Compare performance by model
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='model', y=metric)
    plt.title(f'Model Performance Comparison ({metric})')
    plt.xticks(rotation=45)
    
    # Compare preprocessing effects
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='preprocessing', y=metric)
    plt.title(f'Preprocessing Effect on {metric}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def generate_performance_report(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Analyzes evaluation results, generates a performance report, and saves it as CSV."""
    report_list = []
    
    # Model method mapping
    model_method_map = {
        'base_tesseract': {'text_detection': 'tesseract', 'text_recognition': 'tesseract'},
        'base_easyocr': {'text_detection': 'easyocr', 'text_recognition': 'easyocr'},
        'base_paddleocr': {'text_detection': 'paddleocr', 'text_recognition': 'paddleocr'},
        'base_yolo_ocr': {'text_detection': 'yolo', 'text_recognition': 'yolo'},
        'base_azure_read': {'text_detection': 'azure', 'text_recognition': 'azure'}
    }
    
    for config_key, result_data in all_results.items():
        config = result_data['config']
        metrics = result_data['metrics']
        
        # Extract base model information from model_name
        model_name = config['model_name']
        method_info = model_method_map.get(model_name, {'text_detection': model_name, 'text_recognition': model_name})
        if model_name == "base_easyocr":
            model_name_out = "easyocr"
        elif model_name.startswith('base_'):
            model_name_out = model_name.split('_', 1)[1]
        else:
            model_name_out = model_name
        
        row = {
            'model_name': model_name_out,
            'text_detection': method_info['text_detection'],
            'text_recognition': method_info['text_recognition'],
            'preprocessing_steps': "_".join(config['preprocessing_steps']) if config['preprocessing_steps'] else 'no_preprocessing',
            'item_accuracy': metrics.get('item_accuracy', 0),
            'char_accuracy': metrics.get('char_accuracy', 0),
            'inference_time': metrics.get('inference_time', 0),
        }
        
        # Add detailed metrics
        for metric_type in ['type', 'region', 'length', 'size']:
            metric_key = f'{metric_type}_accuracies'
            if metric_key in metrics:
                for k, v in metrics[metric_key].items():
                    row[f'average_{metric_type}_{k}'] = v

        # Add Text Similarity metrics
        if 'text_similarity' in metrics:
            ts_metrics = metrics['text_similarity']
            row['average_normalized_levenshtein'] = ts_metrics.get('normalized_levenshtein', 0)
            row['average_bleu_score'] = ts_metrics.get('bleu_score', 0)
            if 'rouge_scores' in ts_metrics:
                for rouge_type, score in ts_metrics['rouge_scores'].items():
                    row[f'average_rouge_{rouge_type}'] = score
        
        # Add Full Text Accuracy metrics
        if 'full_text_accuracy' in metrics:
            ft_metrics = metrics['full_text_accuracy']
            row['full_text_exact_match'] = ft_metrics.get('exact_match', 0)
            row['full_text_char_accuracy'] = ft_metrics.get('char_accuracy', 0)
            row['full_text_word_accuracy'] = ft_metrics.get('word_accuracy', 0)
            row['full_text_normalized_accuracy'] = ft_metrics.get('normalized_accuracy', 0)
        
        report_list.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(report_list)
    timestamp = datetime.now().strftime("%Y%m%d")
    csv_path = f"results/{timestamp}_performance_report_{len(report_list)}.csv"
    df.to_csv(csv_path, index=False)
    
    return df 