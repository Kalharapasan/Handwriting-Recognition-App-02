import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import base64

from database import db_manager
from utils import ImagePreprocessor, model_manager

st.set_page_config(
    page_title="Advanced Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    .stat-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
def init_session_state():
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None

def save_uploaded_file(uploaded_file, file_type):
    os.makedirs(f"uploaded_files/{file_type}", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = uploaded_file.name.split('.')[-1]
    file_path = f"uploaded_files/{file_type}/{timestamp}.{file_ext}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def plot_prediction_probabilities(predictions):
    digits = list(range(10))
    probabilities = predictions[0]
    
    fig = go.Figure(data=[
        go.Bar(x=digits, y=probabilities,
               marker_color=['red' if i == np.argmax(probabilities) else 'blue' for i in digits])
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Digits",
        yaxis_title="Probability",
        showlegend=False
    )
    
    return fig

def main():
    local_css()
    init_session_state()
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
        ["🏠 Dashboard", "✍️ Draw Digit", "📁 Upload Image", "📄 Upload Document", "📊 Analytics", "⚙️ Model Management"])