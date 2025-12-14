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
    user_input_type = Column(String(50))  # 'drawing', 'upload', 'document'
    file_name = Column(String(255))

class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_id = Column(Integer)
    actual_digit = Column(Integer)
    correct_prediction = Column(Integer)  # 1 for correct, 0 for incorrect
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
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_prediction(self, predicted_digit, confidence, image_path, user_input_type, file_name):
        """Add a new prediction to the database"""
        try:
            prediction = PredictionHistory(
                predicted_digit=predicted_digit,
                confidence=confidence,
                image_path=image_path,
                user_input_type=user_input_type,
                file_name=file_name
            )
            self.session.add(prediction)
            self.session.commit()
            return prediction.id
        except Exception as e:
            print(f"Error adding prediction: {e}")
            self.session.rollback()
            return None
    
    def add_feedback(self, prediction_id, actual_digit, correct_prediction, comments=""):
        """Add user feedback for a prediction"""
        try:
            feedback = UserFeedback(
                prediction_id=prediction_id,
                actual_digit=actual_digit,
                correct_prediction=correct_prediction,
                comments=comments
            )
            self.session.add(feedback)
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error adding feedback: {e}")
            self.session.rollback()
            return False
    
    def add_model_performance(self, accuracy, loss, validation_accuracy, validation_loss):
        """Add model performance metrics"""
        try:
            performance = ModelPerformance(
                accuracy=accuracy,
                loss=loss,
                validation_accuracy=validation_accuracy,
                validation_loss=validation_loss
            )
            self.session.add(performance)
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error adding performance: {e}")
            self.session.rollback()
            return False
    
    def get_prediction_history(self, limit=100):
        """Get recent prediction history"""
        try:
            return self.session.query(PredictionHistory)\
                       .order_by(PredictionHistory.timestamp.desc())\
                       .limit(limit)\
                       .all()
        except Exception as e:
            print(f"Error getting prediction history: {e}")
            return []
    
    def get_all_predictions(self):
        """Get all predictions"""
        try:
            return self.session.query(PredictionHistory).all()
        except Exception as e:
            print(f"Error getting all predictions: {e}")
            return []
    
    def get_feedback_history(self, limit=100):
        """Get recent feedback"""
        try:
            return self.session.query(UserFeedback)\
                       .order_by(UserFeedback.timestamp.desc())\
                       .limit(limit)\
                       .all()
        except Exception as e:
            print(f"Error getting feedback: {e}")
            return []
    
    def get_performance_stats(self):
        """Calculate and return performance statistics"""
        try:
            # Calculate accuracy from feedback
            feedbacks = self.session.query(UserFeedback).all()
            if feedbacks:
                correct = sum(1 for f in feedbacks if f.correct_prediction == 1)
                total = len(feedbacks)
                user_accuracy = correct / total if total > 0 else 0.0
            else:
                user_accuracy = 0.0
            
            # Get recent predictions
            recent_predictions = self.session.query(PredictionHistory)\
                                     .order_by(PredictionHistory.timestamp.desc())\
                                     .limit(50)\
                                     .all()
            
            # Calculate average confidence
            if recent_predictions:
                confidences = [p.confidence for p in recent_predictions]
                average_confidence = float(np.mean(confidences))
            else:
                average_confidence = 0.0
            
            return {
                'user_accuracy': user_accuracy,
                'total_predictions': len(recent_predictions),
                'average_confidence': average_confidence
            }
        except Exception as e:
            print(f"Error calculating performance stats: {e}")
            return {
                'user_accuracy': 0.0,
                'total_predictions': 0,
                'average_confidence': 0.0
            }
    
    def get_digit_statistics(self):
        """Get statistics grouped by digit"""
        try:
            predictions = self.session.query(PredictionHistory).all()
            if not predictions:
                return {}
            
            digit_stats = {}
            for digit in range(10):
                digit_preds = [p for p in predictions if p.predicted_digit == digit]
                if digit_preds:
                    confidences = [p.confidence for p in digit_preds]
                    digit_stats[digit] = {
                        'count': len(digit_preds),
                        'avg_confidence': float(np.mean(confidences)),
                        'min_confidence': float(np.min(confidences)),
                        'max_confidence': float(np.max(confidences)),
                        'std_confidence': float(np.std(confidences))
                    }
            
            return digit_stats
        except Exception as e:
            print(f"Error getting digit statistics: {e}")
            return {}
    
    def clear_old_data(self, days=30):
        """Clear data older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old predictions
            self.session.query(PredictionHistory)\
                .filter(PredictionHistory.timestamp < cutoff_date)\
                .delete()
            
            # Delete old feedback
            self.session.query(UserFeedback)\
                .filter(UserFeedback.timestamp < cutoff_date)\
                .delete()
            
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error clearing old data: {e}")
            self.session.rollback()
            return False
    
    def export_to_csv(self, output_dir='exports'):
        """Export all data to CSV files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export predictions
            predictions = self.get_all_predictions()
            if predictions:
                pred_data = [{
                    'id': p.id,
                    'timestamp': p.timestamp,
                    'predicted_digit': p.predicted_digit,
                    'confidence': p.confidence,
                    'image_path': p.image_path,
                    'user_input_type': p.user_input_type,
                    'file_name': p.file_name
                } for p in predictions]
                
                df_pred = pd.DataFrame(pred_data)
                df_pred.to_csv(f'{output_dir}/predictions.csv', index=False)
            
            # Export feedback
            feedbacks = self.get_feedback_history(limit=10000)
            if feedbacks:
                feedback_data = [{
                    'id': f.id,
                    'timestamp': f.timestamp,
                    'prediction_id': f.prediction_id,
                    'actual_digit': f.actual_digit,
                    'correct_prediction': f.correct_prediction,
                    'comments': f.comments
                } for f in feedbacks]
                
                df_feedback = pd.DataFrame(feedback_data)
                df_feedback.to_csv(f'{output_dir}/feedback.csv', index=False)
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def get_summary(self):
        """Get a summary of database contents"""
        try:
            total_predictions = self.session.query(PredictionHistory).count()
            total_feedback = self.session.query(UserFeedback).count()
            
            # Get date range
            first_prediction = self.session.query(PredictionHistory)\
                                   .order_by(PredictionHistory.timestamp.asc())\
                                   .first()
            last_prediction = self.session.query(PredictionHistory)\
                                  .order_by(PredictionHistory.timestamp.desc())\
                                  .first()
            
            summary = {
                'total_predictions': total_predictions,
                'total_feedback': total_feedback,
                'first_prediction_date': first_prediction.timestamp if first_prediction else None,
                'last_prediction_date': last_prediction.timestamp if last_prediction else None
            }
            
            return summary
        except Exception as e:
            print(f"Error getting summary: {e}")
            return {}
    
    def close(self):
        """Close database session"""
        try:
            self.session.close()
        except Exception as e:
            print(f"Error closing session: {e}")

# Initialize database manager
db_manager = DatabaseManager()

# Print database summary on import (useful for debugging)
if __name__ == "__main__":
    print("Database Summary:")
    print("-" * 50)
    summary = db_manager.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("-" * 50)
