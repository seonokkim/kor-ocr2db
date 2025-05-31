from .base import BaseOCRModel
from .easyocr import EasyOCRModel
from .yolo_ocr import YOLOOCRModel
from .tesseract import TesseractModel

__all__ = [
    'BaseOCRModel',
    'EasyOCRModel',
    'YOLOOCRModel',
    'TesseractModel',
]

# PaddleOCR model is loaded optionally
try:
    from .paddleocr import PaddleOCRModel
    __all__.append('PaddleOCRModel')
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"\nWarning: Could not load PaddleOCR model - {str(e)}")
    print("PaddleOCR model will be excluded from evaluation.")
    PADDLEOCR_AVAILABLE = False

# 모델 로더 함수 등 필요한 유틸리티 임포트
# from .model_loader import load_model
