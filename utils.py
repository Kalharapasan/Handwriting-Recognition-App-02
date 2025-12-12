import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import os
from pdf2image import convert_from_path
import tempfile

class ImagePreprocessor:
    
    @staticmethod
    def extract_digits_from_image(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def convert_pdf_to_images(pdf_path):
        images = convert_from_path(pdf_path)
        return images
    
    @staticmethod
    def enhance_image_quality(image):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image

class ModelManager:
    def __init__(self, model_path=None):
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            self.model = self.create_default_model()
    
    def create_default_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
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