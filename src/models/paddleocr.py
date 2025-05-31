from typing import List, Tuple
import numpy as np
from paddleocr import PaddleOCR
from .base import BaseOCRModel
import logging

class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR 엔진을 사용하는 모델"""
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        super().__init__(config_path)
        try:
            # 한글 인식을 위한 PaddleOCR 초기화
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='korean'
            )
            logging.info("PaddleOCR initialized with Korean language support")
        except Exception as e:
            logging.error("Error initializing PaddleOCR: %s", str(e))
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # PaddleOCR은 BGR 이미지를 사용하므로 별도 전처리 없음
        return image

    def predict(self, processed_image: np.ndarray):
        try:
            # PaddleOCR로 텍스트 추출
            result = self.ocr.ocr(processed_image, cls=True)
            if result is None or len(result) == 0:
                logging.warning("No text detected in image")
                return []
            
            # 결과 형식 변환: [(text, confidence), [x1, y1, x2, y2]]
            predictions = []
            for line in result[0]:
                text = line[1][0]  # 텍스트
                confidence = line[1][1]  # 신뢰도
                box = line[0]  # 바운딩 박스
                
                # 바운딩 박스를 [x1, y1, x2, y2] 형식으로 변환
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                
                predictions.append((text, bbox))
            
            logging.debug("PaddleOCR predictions: %s", predictions)
            return predictions
        except Exception as e:
            logging.error("Error in PaddleOCR prediction: %s", str(e))
            return []

    def postprocess(self, prediction_result):
        return prediction_result

    # __call__ method is inherited from BaseOCRModel 