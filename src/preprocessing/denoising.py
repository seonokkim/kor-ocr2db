import cv2
import numpy as np
from .base import BasePreprocessor

class DenoisingPreprocessor(BasePreprocessor):
    def __init__(self, h: int = 10, template_window_size: int = 7, search_window_size: int = 21):
        self.h = h  # Filter strength
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply noise removal."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        
        # Convert back to original format
        if len(image.shape) == 3:
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return denoised 