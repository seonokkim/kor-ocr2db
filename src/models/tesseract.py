import numpy as np
import pytesseract
from .base import BaseOCRModel
import logging

class TesseractModel(BaseOCRModel):
    """Tesseract OCR 엔진을 사용하는 모델"""
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        super().__init__(config_path)
        # 한글 언어 데이터 확인
        try:
            pytesseract.get_tesseract_version()
            logging.info("Tesseract version: %s", pytesseract.get_tesseract_version())
            # 사용 가능한 언어 목록 확인
            langs = pytesseract.get_languages()
            logging.info("Available languages: %s", langs)
            if 'kor' not in langs:
                logging.warning("Korean language data not found in Tesseract!")
        except Exception as e:
            logging.error("Error initializing Tesseract: %s", str(e))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Tesseract는 BGR 이미지를 사용하므로 별도 전처리 없음
        return image

    def predict(self, processed_image: np.ndarray):
        try:
            # pytesseract로 전체 이미지에서 텍스트 추출
            text = pytesseract.image_to_string(processed_image, lang='kor')
            h, w = processed_image.shape[:2]
            # 전체 이미지를 하나의 박스로 간주
            result = [(text.strip(), [0.0, 0.0, float(w), float(h)])]
            logging.debug("Tesseract prediction: %s", result)
            return result
        except Exception as e:
            logging.error("Error in Tesseract prediction: %s", str(e))
            return [("", [0.0, 0.0, 0.0, 0.0])]

    def postprocess(self, prediction_result):
        # 이미 predict에서 (text, [x1, y1, x2, y2]) 형태로 반환
        return prediction_result 