from abc import ABC, abstractmethod
import yaml
from pathlib import Path

class BaseOCRModel(ABC):
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = "cpu"  # Azure OCR runs on cloud, so we don't need GPU
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @abstractmethod
    def preprocess(self, image):
        """Preprocess the input image"""
        pass
    
    @abstractmethod
    def predict(self, image):
        """Perform OCR prediction on the image"""
        pass
    
    @abstractmethod
    def postprocess(self, prediction):
        """Postprocess the model output"""
        pass
    
    def __call__(self, image):
        """Run the complete OCR pipeline"""
        preprocessed = self.preprocess(image)
        prediction = self.predict(preprocessed)
        return self.postprocess(prediction) 