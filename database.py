import sqlite3
import pandas as pd
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