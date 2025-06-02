from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from typing import List, Tuple
import numpy as np
import cv2

class AzureDocumentIntelligenceModel:
    def __init__(self, endpoint: str = None, key: str = None):
        """Initialize Azure Document Intelligence client.
        
        Args:
            endpoint (str): Azure Document Intelligence endpoint URL
            key (str): Azure Document Intelligence API key
        """
        self.endpoint = endpoint or os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
        self.key = key or os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
        
        if not self.endpoint or not self.key:
            raise ValueError("Azure Document Intelligence endpoint and key must be provided")
            
        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
    
    def __call__(self, image: np.ndarray) -> List[Tuple[str, List[int]]]:
        """Extract text and bounding boxes from image using Azure Document Intelligence.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Tuple[str, List[int]]]: List of (text, bbox) tuples
                bbox format: [x1, y1, x2, y2]
        """
        # Convert numpy array to bytes
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        # Analyze document
        poller = self.client.begin_analyze_document(
            "prebuilt-document", img_bytes
        )
        result = poller.result()
        
        # Extract text and bounding boxes
        predictions = []
        for page in result.pages:
            for line in page.lines:
                # Get bounding box coordinates
                bbox = [
                    line.bounding_polygon[0].x,  # x1
                    line.bounding_polygon[0].y,  # y1
                    line.bounding_polygon[2].x,  # x2
                    line.bounding_polygon[2].y   # y2
                ]
                predictions.append((line.content, bbox))
                
        return predictions 