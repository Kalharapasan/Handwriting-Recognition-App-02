import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import os
from pdf2image import convert_from_path
import tempfile

class ModelManager:
    
    def predict_digit(self, image):
        if self.model is None:
            return 0, 0.0
        
        predictions = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_digit, confidence
    
    def predict_multiple_digits(self, images):
        results = []
        for image in images:
            processed_image = ImagePreprocessor.preprocess_image(image)
            digit, confidence = self.predict_digit(processed_image)
            results.append((digit, confidence))
        return results