import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_digit = Column(Integer)
    confidence = Column(Float)
    image_path = Column(String(500))
    user_input_type = Column(String(50))  
    file_name = Column(String(255))

class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_id = Column(Integer)
    actual_digit = Column(Integer)
    correct_prediction = Column(Integer)  
    comments = Column(Text)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    loss = Column(Float)
    validation_accuracy = Column(Float)
    validation_loss = Column(Float)


class DatabaseManager: 
    
    def __init__(self, db_path='handwriting_db.sqlite'):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def add_prediction(self, predicted_digit, confidence, image_path, user_input_type, file_name):
        with self.SessionLocal() as session:
            prediction = PredictionHistory(
                predicted_digit=predicted_digit,
                confidence=confidence,
                image_path=image_path,
                user_input_type=user_input_type,
                file_name=file_name
            )
            session.add(prediction)
            session.commit()
            return prediction.id

    def add_feedback(self, prediction_id, actual_digit, correct_prediction, comments=""):
        with self.SessionLocal() as session:
            feedback = UserFeedback(
                prediction_id=prediction_id,
                actual_digit=actual_digit,
                correct_prediction=correct_prediction,
                comments=comments
            )
            session.add(feedback)
            session.commit()
        
    def get_prediction_history(self, limit=100):
        with self.SessionLocal() as session:
            return session.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()

    def get_performance_stats(self):
        with self.SessionLocal() as session:
            feedbacks = session.query(UserFeedback).all()
            if feedbacks:
                correct = sum(1 for f in feedbacks if f.correct_prediction == 1)
                total = len(feedbacks)
                accuracy = correct / total if total > 0 else 0
            else:
                accuracy = 0
            recent_predictions = session.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(50).all()
            
            return {
                'user_accuracy': accuracy,
                'total_predictions': len(recent_predictions),
                'average_confidence': np.mean([p.confidence for p in recent_predictions]) if recent_predictions else 0
            }
db_manager = DatabaseManager()