import yaml
import os
import cv2 # Import OpenCV
import numpy as np # Import numpy
from paddleocr import PaddleOCR

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def apply_image_processing(image, steps):
    processed_image = image.copy()
    for step in steps:
        if step == 'sharpening':
            print("Applying sharpening...")
            # Simple sharpening kernel
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)
        # Add other processing steps here (e.g., denoising, thresholding)
        else:
            print(f"Warning: Unknown image processing step: {step}")
    return processed_image

def main():
    config = load_config()
    use_gpu = config.get('use_gpu', False)
    input_image_path = config.get('input_image_path')
    image_processing_steps = config.get('image_processing_steps', []) # Read processing steps
    model_path = config.get('model_path') # Read model path

    if not input_image_path or not os.path.exists(input_image_path):
        print(f"Error: Input image path not found or does not exist: {input_image_path}")
        return

    print(f"GPU usage enabled: {use_gpu}")
    if image_processing_steps:
        print(f"Applying image processing steps: {image_processing_steps}")

    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Apply image processing if steps are defined
    if image_processing_steps:
        image = apply_image_processing(image, image_processing_steps)

    # Initialize PaddleOCR
    # If model_path is specified, use it. Otherwise, use default.
    if model_path:
        print(f"Using custom model from: {model_path}")
        # TODO: Initialize PaddleOCR with custom model paths
        # PaddleOCR requires paths to different model components (rec_model_dir, det_model_dir, cls_model_dir)
        # This part needs adjustment based on how your custom model is structured.
        # For now, we'll stick to the default initialization if model_path is given but not fully implemented.
        print("Warning: Custom model loading is not fully implemented yet. Using default model.")
        ocr = PaddleOCR(lang='ko', use_gpu=use_gpu)
    else:
        print("Using default PaddleOCR model.")
        ocr = PaddleOCR(lang='ko', use_gpu=use_gpu)

    # Perform OCR on the image (pass the image array directly)
    result = ocr.ocr(image, cls=True)

    # Print OCR results
    if result and result[0]:
        print("OCR Results:")
        for line in result[0]:
            text, score = line
            print(f"Text: {text[0]}, Score: {score:.2f}")
    else:
        print("No text detected.")

if __name__ == "__main__":
    main() 