# 🎨 Advanced Handwriting Recognition System

A comprehensive handwriting recognition system built with TensorFlow, Streamlit, and Computer Vision. This application uses deep learning to recognize handwritten digits with high accuracy.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎨 **Interactive Drawing Canvas**: Draw digits directly in the browser
- 📁 **Image Upload**: Upload images containing single or multiple digits
- 📄 **Document Processing**: Process PDF documents and extract digits
- 📊 **Advanced Analytics**: View detailed performance metrics and statistics
- 🗄️ **Database Integration**: Track predictions and user feedback
- 🔄 **Real-time Processing**: Instant predictions with confidence scores
- 📈 **Visualization**: Beautiful charts and graphs for insights
- ⚙️ **Model Management**: Upload and manage custom models

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) poppler-utils for PDF processing

### Installation

1. **Clone or download the project:**
```bash
# If you have git
git clone <your-repo-url>
cd handwriting-recognition

# Or simply extract the downloaded zip file
```

2. **Create a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **For PDF support (optional):**
```bash
# On Ubuntu/Debian
sudo apt-get install poppler-utils

# On macOS (with Homebrew)
brew install poppler

# On Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### Training the Model

Before using the application, you need to train a model:

```bash
python handwriting_model.py
```

This will:
- Download the MNIST dataset
- Train a CNN model
- Save the trained model to `models/handwriting_model.h5`
- Generate training visualizations
- Create performance reports

**Training options:**
- **Enhanced Model**: Higher accuracy (~99%), slower training (~30-50 epochs)
- **Simple Model**: Good accuracy (~98%), faster training (~10-20 epochs)

### Running the Application

Once the model is trained, start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📂 Project Structure

```
handwriting_recognition_project/
│
├── app.py                      # Main Streamlit application
├── handwriting_model.py        # Model training script
├── database.py                 # Database management
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Trained models
│   ├── handwriting_model.h5
│   ├── handwriting_model.keras
│   └── model_config.json
│
├── uploaded_files/             # User uploaded files
│   ├── images/
│   ├── documents/
│   └── drawings/
│
├── logs/                       # Training logs
│   └── training_history.csv
│
├── plots/                      # Training visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── sample_predictions.png
│
└── templates/                  # HTML templates (optional)
    └── base.html
```

## 🎯 Usage Guide

### 1. Dashboard
- View overall statistics
- See recent predictions
- Quick access to all features

### 2. Draw Digit
- Use the interactive canvas to draw digits
- Get instant predictions with confidence scores
- Provide feedback on predictions

### 3. Upload Image
- Upload images containing digits
- Process single or multiple digits
- View processed images and predictions

### 4. Upload Document
- Upload PDF or text files
- Extract and recognize digits from documents
- Process multiple pages

### 5. Analytics
- View prediction distribution
- Analyze confidence levels
- Track performance over time
- Compare input methods

### 6. Model Management
- View current model details
- Upload custom trained models
- Run test predictions

## 🔧 Configuration

### Database Configuration
The application uses SQLite by default. You can modify `database.py` to use other databases:

```python
# For MySQL
self.engine = create_engine('mysql+pymysql://user:password@localhost/dbname')

# For PostgreSQL
self.engine = create_engine('postgresql://user:password@localhost/dbname')
```

### Model Configuration
Modify `handwriting_model.py` to customize:
- Number of epochs
- Batch size
- Learning rate
- Model architecture
- Data augmentation parameters

## 📊 Model Performance

The trained model achieves:
- **Test Accuracy**: ~98-99%
- **Training Time**: 10-50 minutes (depending on hardware)
- **Model Size**: ~5-10 MB
- **Inference Time**: <100ms per image

### Per-Digit Performance
The model performs exceptionally well across all digits (0-9), with individual class accuracies typically above 97%.

## 🛠️ Troubleshooting

### Common Issues

**Issue: Model file not found**
```bash
Solution: Run python handwriting_model.py to train the model first
```

**Issue: streamlit-drawable-canvas not working**
```bash
Solution: Reinstall the package
pip uninstall streamlit-drawable-canvas
pip install streamlit-drawable-canvas
```

**Issue: PDF processing fails**
```bash
Solution: Install poppler-utils (see Installation section)
```

**Issue: TensorFlow GPU not detected**
```bash
Solution: Install tensorflow-gpu and ensure CUDA is properly configured
pip install tensorflow-gpu
```

**Issue: Import errors**
```bash
Solution: Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

## 📈 Future Enhancements

- [ ] Support for uppercase and lowercase letters
- [ ] Real-time video digit recognition
- [ ] Mobile app version
- [ ] Batch processing for multiple files
- [ ] Advanced data augmentation
- [ ] Model ensemble for improved accuracy
- [ ] Export predictions to CSV/Excel
- [ ] User authentication and profiles
- [ ] Cloud deployment support
- [ ] API endpoints for integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 [License](./LICENSE.md): Proprietary – Permission Required

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun and Corinna Cortes
- **TensorFlow**: Google Brain Team
- **Streamlit**: Streamlit Team
- **OpenCV**: Intel Corporation and contributors

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**
