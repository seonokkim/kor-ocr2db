import numpy as np
from ultralytics import YOLO
import easyocr
from .base import BaseOCRModel
import logging

class YOLOOCRModel(BaseOCRModel):
    """YOLO 기반 텍스트 검출 + EasyOCR 인식기 조합 모델"""
    def __init__(self, use_gpu: bool = True, config_path: str = "configs/default_config.yaml", yolo_model_path: str = "yolov8n.pt"):
        super().__init__(config_path)
        self.device = "cuda" if use_gpu else "cpu"
        self.yolo = YOLO(yolo_model_path)
        self.ocr = easyocr.Reader(['ko'], gpu=use_gpu)
        self.confidence_threshold = 0.1  # 신뢰도 임계값을 더 낮춤
        self.iou_threshold = 0.2  # IoU 임계값을 더 낮춤

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # YOLO와 EasyOCR 모두 BGR 이미지를 사용하므로 별도 전처리 없음
        return image

    def _combine_text_regions(self, boxes, texts, confidences):
        """인접한 텍스트 영역을 결합"""
        if not boxes:
            return []
        
        # x 좌표 기준으로 정렬
        sorted_indices = np.argsort([box[0] for box in boxes])
        boxes = [boxes[i] for i in sorted_indices]
        texts = [texts[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        
        combined_results = []
        current_group = []
        
        for i in range(len(boxes)):
            if not current_group:
                current_group = [(boxes[i], texts[i], confidences[i])]
                continue
                
            # 현재 박스와 이전 그룹의 마지막 박스 비교
            last_box = current_group[-1][0]
            current_box = boxes[i]
            
            # x 좌표가 가까우면 같은 그룹으로 (임계값 조정)
            if current_box[0] - last_box[2] < self.iou_threshold * (current_box[2] - current_box[0]):
                current_group.append((current_box, texts[i], confidences[i]))
            else:
                # 그룹 결합
                if current_group:
                    combined_box = self._merge_boxes([b[0] for b in current_group])
                    combined_text = ' '.join([t[1] for t in current_group])  # 공백으로 구분
                    avg_confidence = sum([c[2] for c in current_group]) / len(current_group)
                    combined_results.append((combined_text, combined_box, avg_confidence))
                current_group = [(current_box, texts[i], confidences[i])]
        
        # 마지막 그룹 처리
        if current_group:
            combined_box = self._merge_boxes([b[0] for b in current_group])
            combined_text = ' '.join([t[1] for t in current_group])  # 공백으로 구분
            avg_confidence = sum([c[2] for c in current_group]) / len(current_group)
            combined_results.append((combined_text, combined_box, avg_confidence))
        
        return combined_results

    def _merge_boxes(self, boxes):
        """여러 박스를 하나로 병합"""
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [x1, y1, x2, y2]

    def predict(self, processed_image: np.ndarray):
        try:
            # 1. YOLO로 텍스트 영역 검출
            results = self.yolo.predict(processed_image, device=self.device, verbose=False, conf=0.1)  # 신뢰도 임계값 낮춤
            
            # Check if results is empty or None
            if not results or len(results) == 0:
                logging.warning("No results from YOLO prediction")
                return []
                
            # Get boxes and confidences, handling empty cases
            boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else np.array([])
            confidences = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else np.array([])
            
            # Check if boxes array is empty
            if boxes.size == 0:
                logging.warning("No text regions detected by YOLO")
                return []
            
            # 2. 각 박스별로 EasyOCR 인식
            texts = []
            valid_boxes = []
            valid_confidences = []
            
            for box, conf in zip(boxes, confidences):
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                # 박스가 이미지 경계를 벗어나지 않도록 조정
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(processed_image.shape[1], x2)
                y2 = min(processed_image.shape[0], y2)
                # 박스가 너무 작으면 건너뛰기 (크기 제한 완화)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                # 박스가 너무 크면 건너뛰기 (이미지 면적의 50% 이상)
                img_area = processed_image.shape[0] * processed_image.shape[1]
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > img_area * 0.5:
                    continue
                
                crop = processed_image[y1:y2, x1:x2]
                
                # EasyOCR로 텍스트 인식
                ocr_results = self.ocr.readtext(crop)
                if ocr_results:
                    # 가장 신뢰도 높은 결과 사용
                    ocr_results.sort(key=lambda x: x[2], reverse=True)
                    text = ocr_results[0][1]
                    if text.strip():  # 빈 텍스트가 아닌 경우만 추가
                        texts.append(text)
                        valid_boxes.append(box)
                        valid_confidences.append(conf)
            
            # 3. 인접한 텍스트 영역 결합
            combined_results = self._combine_text_regions(valid_boxes, texts, valid_confidences)
            
            # 4. 최종 결과 형식 변환
            def to_py_type_box(box):
                # Convert all elements to int (or float if needed)
                return [int(x) if int(x) == x else float(x) for x in box]
            predictions = [(text, to_py_type_box(box)) for text, box, _ in combined_results]
            
            logging.debug(f"YOLO OCR predictions: {predictions}")
            return predictions
            
        except Exception as e:
            logging.error(f"Error in YOLO OCR prediction: {str(e)}")
            return []

    def postprocess(self, prediction_result):
        # 이미 predict에서 (text, [x1, y1, x2, y2]) 형태로 반환
        return prediction_result 