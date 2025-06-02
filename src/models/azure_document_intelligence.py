from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import logging
from .base import BaseOCRModel

class AzureDocumentIntelligenceModel(BaseOCRModel):
    """Azure Document Intelligence 기반 OCR 모델"""
    
    def __init__(self, mode: str = 'read', use_gpu: bool = True, config_path: str = "configs/default_config.yaml"):
        """
        Args:
            mode (str): 'read', 'layout', or 'prebuilt_read'
            use_gpu (bool): GPU 사용 여부 (Azure는 서버에서 처리하므로 무시됨)
            config_path (str): 설정 파일 경로
        """
        super().__init__(config_path)
        
        # Azure Document Intelligence 설정
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not endpoint or not key:
            raise ValueError("Azure Document Intelligence credentials not found in environment variables")
        
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        self.mode = mode
        if mode not in ['read', 'layout', 'prebuilt_read']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['read', 'layout', 'prebuilt_read']")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Azure Document Intelligence는 이미지 전처리가 필요 없음"""
        return image

    def predict(self, processed_image: np.ndarray) -> List[Tuple[str, List[float]]]:
        try:
            # 이미지를 바이트로 변환
            _, img_encoded = cv2.imencode('.jpg', processed_image)
            image_bytes = img_encoded.tobytes()
            
            # 모드에 따라 다른 분석 수행
            if self.mode == 'read':
                poller = self.client.begin_analyze_document("prebuilt-read", image_bytes)
            elif self.mode == 'layout':
                poller = self.client.begin_analyze_document("prebuilt-layout", image_bytes)
            else:  # prebuilt_read
                poller = self.client.begin_analyze_document("prebuilt-document", image_bytes)
            
            result = poller.result()
            
            predictions = []
            
            # 텍스트와 바운딩 박스 추출
            for page in result.pages:
                for line in page.lines:
                    # 바운딩 박스 좌표 추출
                    points = line.bounding_polygon
                    if points and len(points) >= 4:
                        x_coords = [p.x for p in points]
                        y_coords = [p.y for p in points]
                        bbox = [
                            min(x_coords),  # x1
                            min(y_coords),  # y1
                            max(x_coords),  # x2
                            max(y_coords)   # y2
                        ]
                        predictions.append((line.content, bbox))
            
            logging.debug(f"Azure Document Intelligence predictions: {predictions}")
            return predictions
            
        except Exception as e:
            logging.error(f"Error in Azure Document Intelligence prediction: {str(e)}")
            return []

    def postprocess(self, prediction_result):
        """이미 predict에서 (text, [x1, y1, x2, y2]) 형태로 반환"""
        return prediction_result 