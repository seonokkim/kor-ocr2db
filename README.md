# kor-ocr2db

Korean OCR to Database Project

## Overview

This project aims to perform optical character recognition (OCR) on Korean text from images and store the results, exploring different methods for optimal performance.

## Setup

1.  Clone the repository:

    ```bash
    git clone https://github.com/seonokkim/kor-ocr2db.git
    cd kor-ocr2db
    ```

2.  Install dependencies:

    It's recommended to use a virtual environment.

    ```bash
    # Create a virtual environment (if it doesn't exist, one named 'venv' is expected by default)
    python -m venv venv
    
    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    
    # Install dependencies (using the virtual environment's pip)
    # On Windows:
    venv\Scripts\python.exe -m pip install -r requirements.txt
    # On macOS/Linux:
    # venv/bin/python -m pip install -r requirements.txt
    ```

## Configuration

Edit `config.yaml` to adjust settings.

-   `use_gpu`: Set to `True` to enable GPU acceleration (requires compatible GPU, CUDA, and GPU-enabled libraries). Set to `False` for CPU-only execution.

## Running the Evaluation

To run the OCR evaluation script:

```bash
# On Windows:
venv\Scripts\python.exe run_evaluation.py
# On macOS/Linux:
# venv/bin/python run_evaluation.py
```

## GPU Acceleration

To use GPU acceleration, you need a compatible NVIDIA GPU, CUDA Toolkit, and cuDNN. Install the GPU-compatible versions of libraries like `paddleocr` (e.g., `pip install paddleocr paddlepaddle-gpu`). Refer to the specific library's documentation for detailed GPU installation instructions.

## Models

This project aims to integrate and compare different high-performance OCR models for Korean. Model files should be placed in a designated directory (e.g., `models/`). Details on specific models and how to use them will be added as they are integrated. 