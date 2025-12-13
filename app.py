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
    
    st.markdown('<h1 class="main-header">✍️ Advanced Handwriting Recognition System</h1>', unsafe_allow_html=True)
    
    if app_mode == "🏠 Dashboard":
        show_dashboard()
    elif app_mode == "✍️ Draw Digit":
        show_drawing_interface()
    elif app_mode == "📁 Upload Image":
        show_image_upload()
    elif app_mode == "📄 Upload Document":
        show_document_upload()
    elif app_mode == "📊 Analytics":
        show_analytics()
    elif app_mode == "⚙️ Model Management":
        show_model_management()

def show_dashboard():
    col1, col2, col3 = st.columns(3)
    stats = db_manager.get_performance_stats()
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>📈 User Accuracy</h3>
            <h2>{stats['user_accuracy']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>🔢 Total Predictions</h3>
            <h2>{stats['total_predictions']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>🎯 Avg Confidence</h3>
            <h2>{stats['average_confidence']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Recent Predictions")
    recent_predictions = db_manager.get_prediction_history(limit=10)
    
    if recent_predictions:
        prediction_data = []
        for pred in recent_predictions:
            prediction_data.append({
                'Timestamp': pred.timestamp,
                'Digit': pred.predicted_digit,
                'Confidence': f"{pred.confidence:.1%}",
                'Type': pred.user_input_type,
                'File': pred.file_name
            })
        
        df = pd.DataFrame(prediction_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions yet. Start by drawing or uploading an image!")
    
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 Start Drawing", use_container_width=True):
            st.session_state.current_page = "Draw Digit"
            st.rerun()
    
    with col2:
        if st.button("📁 Upload Image", use_container_width=True):
            st.session_state.current_page = "Upload Image"
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "Analytics"
            st.rerun()

def show_drawing_interface():
    st.subheader("Draw a Digit (0-9)")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
            return
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    with col2:
        st.subheader("Controls")
        
        if st.button("🔍 Predict Drawing", use_container_width=True):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                pil_image = Image.fromarray(img_array.astype('uint8'))
                processed_image = ImagePreprocessor.preprocess_image(np.array(pil_image))
                
                predicted_digit, confidence = model_manager.predict_digit(processed_image)
                file_path = save_uploaded_file_placeholder("drawing", pil_image)
                prediction_id = db_manager.add_prediction(
                    predicted_digit, confidence, file_path, "drawing", "hand_drawn"
                )
                
                st.session_state.current_prediction = {
                    'id': prediction_id,
                    'digit': predicted_digit,
                    'confidence': confidence,
                    'image': processed_image
                }
        if st.button("🧹 Clear Canvas", use_container_width=True):
            st.rerun()
        
        if st.session_state.current_prediction:
            prediction = st.session_state.current_prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Prediction Result</h3>
                <p><strong>Digit:</strong> {prediction['digit']}</p>
                <p><strong>Confidence:</strong> <span class="confidence-high">{prediction['confidence']:.1%}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Provide Feedback")
            actual_digit = st.number_input("Actual Digit", min_value=0, max_value=9, value=0)
            correct = st.radio("Was the prediction correct?", ["Yes", "No"])
            comments = st.text_area("Comments (optional)")
            
            if st.button("Submit Feedback"):
                is_correct = 1 if correct == "Yes" else 0
                db_manager.add_feedback(prediction['id'], actual_digit, is_correct, comments)
                st.success("Thank you for your feedback!")
                st.session_state.current_prediction = None
def show_image_upload():
    st.subheader("Upload Image File")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            process_type = st.radio("Processing Type", 
                                  ["Single Digit", "Multiple Digits"])
        
        with col2:
            if st.button("🔍 Analyze Image", use_container_width=True):
                if process_type == "Single Digit":
                    image_array = np.array(image)
                    processed_image = ImagePreprocessor.preprocess_image(image_array)
                    predicted_digit, confidence = model_manager.predict_digit(processed_image)
                    file_path = save_uploaded_file(uploaded_file, "images")
                    prediction_id = db_manager.add_prediction(
                        predicted_digit, confidence, file_path, "image_upload", uploaded_file.name
                    )
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Prediction Result</h3>
                        <p><strong>Digit:</strong> {predicted_digit}</p>
                        <p><strong>Confidence:</strong> <span class="confidence-high">{confidence:.1%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    ax1.imshow(image_array, cmap='gray')
                    ax1.set_title("Original Image")
                    ax1.axis('off')
                    
                    ax2.imshow(processed_image.reshape(28, 28), cmap='gray')
                    ax2.set_title("Processed Image")
                    ax2.axis('off')
                    
                    st.pyplot(fig)
        
    
        
        
                