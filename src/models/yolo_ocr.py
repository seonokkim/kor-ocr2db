import numpy as np
from ultralytics import YOLO
import easyocr
from .base import BaseOCRModel

class YOLOOCRModel(BaseOCRModel):
    """YOLO 기반 텍스트 검출 + EasyOCR 인식기 조합 모델"""
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml", yolo_model_path: str = "yolov8n.pt"):
        super().__init__(config_path)
        self.device = "cuda" if use_gpu else "cpu"
        self.yolo = YOLO(yolo_model_path)
        self.ocr = easyocr.Reader(['ko'], gpu=use_gpu)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # YOLO와 EasyOCR 모두 BGR 이미지를 사용하므로 별도 전처리 없음
        return image

    def predict(self, processed_image: np.ndarray):
        # 1. YOLO로 텍스트 영역 검출
        results = self.yolo.predict(processed_image, device=self.device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
        # 2. 각 박스별로 EasyOCR 인식
        predictions = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = processed_image[y1:y2, x1:x2]
            ocr_results = self.ocr.readtext(crop)
            if ocr_results:
                # 가장 신뢰도 높은 결과만 사용
                ocr_results.sort(key=lambda x: x[2], reverse=True)
                text = ocr_results[0][1]
                predictions.append((text, [float(x1), float(y1), float(x2), float(y2)]))
        return predictions

    def postprocess(self, prediction_result):
        # 이미 predict에서 (text, [x1, y1, x2, y2]) 형태로 반환
        return prediction_result 