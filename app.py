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

# Import our modules
from database import db_manager
from utils import ImagePreprocessor, model_manager

# Page configuration
st.set_page_config(
    page_title="Advanced Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    """Initialize session state variables"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None

def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file and return path"""
    os.makedirs(f"uploaded_files/{file_type}", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = uploaded_file.name.split('.')[-1]
    file_path = f"uploaded_files/{file_type}/{timestamp}.{file_ext}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def plot_prediction_probabilities(predictions):
    """Plot prediction probabilities using Plotly"""
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
        showlegend=False,
        height=400
    )
    
    return fig

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.9:
        return "confidence-high"
    elif confidence >= 0.7:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    local_css()
    init_session_state()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
        ["🏠 Dashboard", "✍️ Draw Digit", "📁 Upload Image", "📄 Upload Document", "📊 Analytics", "⚙️ Model Management"])
    
    # Main header
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
    """Show main dashboard"""
    col1, col2, col3 = st.columns(3)
    
    # Get performance stats
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
    
    # Recent predictions
    st.subheader("Recent Predictions")
    recent_predictions = db_manager.get_prediction_history(limit=10)
    
    if recent_predictions:
        prediction_data = []
        for pred in recent_predictions:
            prediction_data.append({
                'Timestamp': pred.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'Digit': pred.predicted_digit,
                'Confidence': f"{pred.confidence:.1%}",
                'Type': pred.user_input_type,
                'File': pred.file_name
            })
        
        df = pd.DataFrame(prediction_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions yet. Start by drawing or uploading an image!")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🎨 Draw a digit using the canvas")
    
    with col2:
        st.info("📁 Upload an image for recognition")
    
    with col3:
        st.info("📊 View detailed analytics")

def show_drawing_interface():
    """Show drawing interface using Streamlit's drawable canvas"""
    st.subheader("Draw a Digit (0-9)")
    
    # Check if streamlit-drawable-canvas is available
    try:
        from streamlit_drawable_canvas import st_canvas
        has_canvas = True
    except ImportError:
        has_canvas = False
        st.error("⚠️ streamlit-drawable-canvas is not installed. Please install it using: `pip install streamlit-drawable-canvas`")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if has_canvas:
            # Canvas specifications
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔮 Predict", use_container_width=True, type="primary"):
                    if canvas_result.image_data is not None:
                        # Get the image data
                        image_data = canvas_result.image_data
                        
                        # Check if there's any drawing
                        if np.sum(image_data[:, :, 3]) > 0:  # Check alpha channel
                            # Convert to grayscale
                            gray_image = cv2.cvtColor(image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
                            
                            # Preprocess
                            processed_image = ImagePreprocessor.preprocess_image(gray_image)
                            
                            # Predict
                            predictions = model_manager.model.predict(processed_image, verbose=0)
                            predicted_digit = int(np.argmax(predictions[0]))
                            confidence = float(np.max(predictions[0]))
                            
                            # Save to database
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_path = f"uploaded_files/drawings/drawing_{timestamp}.png"
                            os.makedirs("uploaded_files/drawings", exist_ok=True)
                            cv2.imwrite(file_path, gray_image)
                            
                            prediction_id = db_manager.add_prediction(
                                predicted_digit, confidence, file_path, "drawing", f"drawing_{timestamp}.png"
                            )
                            
                            # Store in session state
                            st.session_state.current_prediction = {
                                'digit': predicted_digit,
                                'confidence': confidence,
                                'predictions': predictions,
                                'id': prediction_id
                            }
                        else:
                            st.warning("Please draw something first!")
            
            with col_b:
                if st.button("🗑️ Clear Canvas", use_container_width=True):
                    st.session_state.current_prediction = None
                    st.rerun()
        else:
            st.info("Canvas drawing requires the streamlit-drawable-canvas package. Using alternative input method.")
            st.text_input("Enter a digit (0-9) for testing:", key="manual_digit")
    
    with col2:
        st.markdown("### Instructions")
        st.write("1. Draw a single digit (0-9)")
        st.write("2. Try to center the digit")
        st.write("3. Use clear strokes")
        st.write("4. Click 'Predict' to recognize")
        
        # Show prediction if available
        if st.session_state.current_prediction:
            pred = st.session_state.current_prediction
            confidence_class = get_confidence_class(pred['confidence'])
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Prediction Result</h3>
                <h1 style="text-align: center; color: #1f77b4;">{pred['digit']}</h1>
                <p><strong>Confidence:</strong> <span class="{confidence_class}">{pred['confidence']:.1%}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability chart
            st.plotly_chart(plot_prediction_probabilities(pred['predictions']), use_container_width=True)
            
            # Feedback section
            st.markdown("### Feedback")
            actual_digit = st.number_input("Was this correct? Enter the actual digit:", 
                                          min_value=0, max_value=9, step=1, key="feedback_digit")
            
            if st.button("Submit Feedback"):
                correct = 1 if actual_digit == pred['digit'] else 0
                db_manager.add_feedback(pred['id'], actual_digit, correct)
                st.success("Thank you for your feedback!")

def show_image_upload():
    """Show image upload interface"""
    st.subheader("Upload Image for Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'bmp'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                process_type = st.radio("Processing Mode", 
                                       ["Single Digit", "Multiple Digits"],
                                       help="Choose how to process the image")
            
            with col_b:
                if st.button("🔮 Analyze Image", use_container_width=True, type="primary"):
                    with st.spinner("Processing image..."):
                        # Convert to numpy array
                        image_array = np.array(image)
                        
                        if process_type == "Single Digit":
                            # Save file
                            file_path = save_uploaded_file(uploaded_file, "images")
                            
                            # Preprocess
                            processed_image = ImagePreprocessor.preprocess_image(image_array)
                            
                            # Predict
                            predictions = model_manager.model.predict(processed_image, verbose=0)
                            predicted_digit = int(np.argmax(predictions[0]))
                            confidence = float(np.max(predictions[0]))
                            
                            # Save to database
                            prediction_id = db_manager.add_prediction(
                                predicted_digit, confidence, file_path, "image_upload", uploaded_file.name
                            )
                            
                            # Display results in col2
                            with col2:
                                confidence_class = get_confidence_class(confidence)
                                
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h3>Prediction Result</h3>
                                    <h1 style="text-align: center; color: #1f77b4;">{predicted_digit}</h1>
                                    <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence:.1%}</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show comparison
                            st.subheader("Image Processing")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            
                            if len(image_array.shape) == 3:
                                ax1.imshow(image_array)
                            else:
                                ax1.imshow(image_array, cmap='gray')
                            ax1.set_title("Original Image")
                            ax1.axis('off')
                            
                            ax2.imshow(processed_image.reshape(28, 28), cmap='gray')
                            ax2.set_title("Processed Image (28x28)")
                            ax2.axis('off')
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            # Show probability distribution
                            st.plotly_chart(plot_prediction_probabilities(predictions), use_container_width=True)
                        
                        else:
                            # Process multiple digits
                            file_path = save_uploaded_file(uploaded_file, "images")
                            digit_images = ImagePreprocessor.extract_digits_from_image(file_path)
                            
                            if len(digit_images) == 0:
                                st.warning("No digits found in the image. Try adjusting the image or using Single Digit mode.")
                            else:
                                st.success(f"Found {len(digit_images)} digit(s) in the image")
                                
                                # Display all predictions
                                for i, digit_img in enumerate(digit_images):
                                    col_x, col_y = st.columns([1, 3])
                                    
                                    with col_x:
                                        st.image(digit_img, caption=f"Digit {i+1}", width=100)
                                    
                                    with col_y:
                                        processed_image = ImagePreprocessor.preprocess_image(digit_img)
                                        predictions = model_manager.model.predict(processed_image, verbose=0)
                                        predicted_digit = int(np.argmax(predictions[0]))
                                        confidence = float(np.max(predictions[0]))
                                        
                                        confidence_class = get_confidence_class(confidence)
                                        st.markdown(f"""
                                        Prediction: **{predicted_digit}** 
                                        (Confidence: <span class="{confidence_class}">{confidence:.1%}</span>)
                                        """, unsafe_allow_html=True)
                                        
                                        # Save each prediction
                                        db_manager.add_prediction(
                                            predicted_digit, confidence, file_path, 
                                            "image_upload_multiple", f"{uploaded_file.name}_digit_{i+1}"
                                        )
                                    
                                    st.markdown("---")
    
    with col2:
        st.markdown("### Tips")
        st.write("📌 Use clear, high-contrast images")
        st.write("📌 Center the digit in the image")
        st.write("📌 Avoid background noise")
        st.write("📌 Single digits work best")

def show_document_upload():
    """Show document upload interface"""
    st.subheader("Upload Document (PDF/Text)")
    
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            st.info("📄 PDF Processing")
            # Handle PDF files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Convert PDF to images
                images = ImagePreprocessor.convert_pdf_to_images(tmp_path)
                st.success(f"PDF converted to {len(images)} page(s)")
                
                # Process each page
                for page_num, image in enumerate(images):
                    with st.expander(f"📄 Page {page_num + 1}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption=f"Page {page_num + 1}", use_container_width=True)
                        
                        with col2:
                            if st.button(f"Analyze Page {page_num + 1}", key=f"page_{page_num}"):
                                # Convert to array and process
                                image_array = np.array(image)
                                processed_image = ImagePreprocessor.preprocess_image(image_array)
                                
                                # Make prediction
                                predictions = model_manager.model.predict(processed_image, verbose=0)
                                predicted_digit = int(np.argmax(predictions[0]))
                                confidence = float(np.max(predictions[0]))
                                
                                # Save image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                file_path = f"uploaded_files/documents/page_{page_num}_{timestamp}.png"
                                os.makedirs("uploaded_files/documents", exist_ok=True)
                                image.save(file_path)
                                
                                # Save to database
                                prediction_id = db_manager.add_prediction(
                                    predicted_digit, confidence, file_path, 
                                    "document", f"{uploaded_file.name}_page_{page_num+1}"
                                )
                                
                                confidence_class = get_confidence_class(confidence)
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h3>Prediction Result</h3>
                                    <h1 style="text-align: center;">{predicted_digit}</h1>
                                    <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence:.1%}</span></p>
                                </div>
                                """, unsafe_allow_html=True)
                
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.info("Make sure poppler is installed for PDF processing. See requirements for details.")
        
        elif file_ext == 'txt':
            # Handle text files
            content = uploaded_file.getvalue().decode()
            st.text_area("File Content", content, height=200)
            
            # Look for digits in text
            digits_found = [char for char in content if char.isdigit()]
            if digits_found:
                st.write(f"Found {len(digits_found)} digit(s) in text:")
                st.write(', '.join(digits_found))
            else:
                st.info("No digits found in the text file.")

def show_analytics():
    """Show analytics and performance metrics"""
    st.subheader("📊 Performance Analytics")
    
    # Get prediction history
    predictions = db_manager.get_prediction_history(limit=1000)
    
    if not predictions:
        st.info("No prediction data available yet. Start making predictions to see analytics!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': p.timestamp,
        'digit': p.predicted_digit,
        'confidence': p.confidence,
        'type': p.user_input_type
    } for p in predictions])
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        st.metric("Average Confidence", f"{df['confidence'].mean():.1%}")
    with col3:
        most_common = int(df['digit'].mode().iloc[0]) if not df.empty else "N/A"
        st.metric("Most Common Digit", most_common)
    with col4:
        days = (df['timestamp'].max() - df['timestamp'].min()).days
        st.metric("Days Active", days if days > 0 else "< 1")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Digit distribution
        st.subheader("Digit Distribution")
        digit_counts = df['digit'].value_counts().sort_index()
        fig = px.bar(x=digit_counts.index, y=digit_counts.values,
                    labels={'x': 'Digit', 'y': 'Count'},
                    title="Frequency of Each Digit")
        fig.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig = px.histogram(df, x='confidence', nbins=20,
                          labels={'confidence': 'Confidence Level'},
                          title="Distribution of Prediction Confidence")
        fig.update_traces(marker_color='#ff7f0e')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("Predictions Over Time")
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df_daily = df.groupby('date').size().reset_index(name='count')
    fig = px.line(df_daily, x='date', y='count',
                 labels={'date': 'Date', 'count': 'Number of Predictions'},
                 title="Daily Prediction Activity")
    fig.update_traces(line_color='#2ecc71')
    st.plotly_chart(fig, use_container_width=True)
    
    # Input type analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Type Distribution")
        type_counts = df['type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                    title="Predictions by Input Method")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confidence by Input Type")
        fig = px.box(df, x='type', y='confidence',
                    labels={'type': 'Input Type', 'confidence': 'Confidence'},
                    title="Confidence Distribution by Input Method")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed statistics table
    st.subheader("Detailed Statistics by Digit")
    digit_stats = df.groupby('digit').agg({
        'confidence': ['mean', 'std', 'min', 'max'],
        'digit': 'count'
    }).round(3)
    digit_stats.columns = ['Avg Confidence', 'Std Dev', 'Min Confidence', 'Max Confidence', 'Count']
    st.dataframe(digit_stats, use_container_width=True)

def show_model_management():
    """Show model management interface"""
    st.subheader("⚙️ Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Model Status")
        if model_manager.model:
            st.success("✅ Model loaded successfully!")
            
            # Display basic model info
            st.info(f"**Model Type:** Convolutional Neural Network (CNN)")
            st.info(f"**Input Shape:** 28x28x1 (grayscale)")
            st.info(f"**Output Classes:** 10 (digits 0-9)")
            
            # Model summary
            with st.expander("View Model Architecture"):
                summary = []
                model_manager.model.summary(print_fn=lambda x: summary.append(x))
                st.text("\n".join(summary))
            
            # Model parameters
            total_params = model_manager.model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")
        else:
            st.error("❌ No model loaded")
    
    with col2:
        st.markdown("### Model Operations")
        
        # Upload new model
        st.write("**Upload New Model**")
        uploaded_model = st.file_uploader("Choose a model file (.h5 or .keras)", 
                                         type=['h5', 'hdf5', 'keras'])
        
        if uploaded_model:
            if st.button("📥 Update Model", type="primary"):
                try:
                    # Save uploaded model
                    os.makedirs("models", exist_ok=True)
                    with open("models/handwriting_model.h5", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    # Reload model
                    model_manager.load_model("models/handwriting_model.h5")
                    st.success("✅ Model updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating model: {str(e)}")
        
        st.markdown("---")
        
        # Test model
        st.write("**Quick Model Test**")
        if st.button("🧪 Run Test Prediction"):
            # Create a random test image
            test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
            test_image = test_image.astype('float32') / 255.0
            test_image = test_image.reshape(1, 28, 28, 1)
            
            # Predict
            predictions = model_manager.model.predict(test_image, verbose=0)
            predicted_digit = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            st.write(f"Test Result: Digit **{predicted_digit}** (Confidence: {confidence:.1%})")
            st.caption("This is a test on random noise - results may vary")
    
    # Performance metrics
    st.markdown("---")
    st.subheader("Performance Metrics")
    
    stats = db_manager.get_performance_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("User Reported Accuracy", f"{stats['user_accuracy']:.1%}")
    with col2:
        st.metric("Total Predictions", stats['total_predictions'])
    with col3:
        st.metric("Average Confidence", f"{stats['average_confidence']:.1%}")
    
    # Additional information
    st.info("""
    **Model Information:**
    - This model is trained on the MNIST dataset
    - Expected accuracy: ~98-99% on standard digits
    - Best performance on clear, centered, single digits
    - May require retraining for specific use cases
    """)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("uploaded_files/images", exist_ok=True)
    os.makedirs("uploaded_files/documents", exist_ok=True)
    os.makedirs("uploaded_files/drawings", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    main()
