# 📦 Installation Guide

Complete installation instructions for the Handwriting Recognition System.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Verify Installation](#verify-installation)
4. [Common Issues](#common-issues)
5. [Platform-Specific Instructions](#platform-specific-instructions)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for initial setup

### Recommended Requirements
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **CPU**: Multi-core processor

## Installation Methods

### Method 1: Automated Setup (Recommended)

**Step 1:** Extract/Download all project files

**Step 2:** Open terminal/command prompt in project directory

**Step 3:** Run the setup script:
```bash
python setup.py
```

This will automatically:
- Check Python version
- Create necessary directories
- Install all dependencies
- Verify installations

### Method 2: Manual Installation

**Step 1:** Install Python
- Download from [python.org](https://www.python.org/downloads/)
- Ensure Python 3.8+ is installed
- Verify: `python --version` or `python3 --version`

**Step 2:** Create Virtual Environment (Recommended)

On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3:** Install Dependencies
```bash
pip install -r requirements.txt
```

**Step 4:** Create Directories
```bash
# On Windows
mkdir models uploaded_files\images uploaded_files\documents uploaded_files\drawings logs plots

# On macOS/Linux
mkdir -p models uploaded_files/{images,documents,drawings} logs plots
```

### Method 3: Conda Environment

If you use Anaconda/Miniconda:

```bash
# Create environment
conda create -n handwriting python=3.9

# Activate environment
conda activate handwriting

# Install packages
pip install -r requirements.txt
```

## Verify Installation

### Quick Test

Run this command to verify everything is installed:
```bash
python -c "import tensorflow, streamlit, cv2, PIL, numpy, pandas; print('✓ All packages imported successfully!')"
```

### Detailed Verification

1. **Check TensorFlow:**
```python
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

2. **Check Streamlit:**
```bash
streamlit --version
```

3. **Check OpenCV:**
```python
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

## Common Issues

### Issue 1: Python Version Too Old

**Error:** `Python 3.8+ is required`

**Solution:**
1. Download latest Python from python.org
2. Install and add to PATH
3. Restart terminal
4. Verify: `python --version`

### Issue 2: pip Not Found

**Error:** `'pip' is not recognized`

**Solution:**
```bash
python -m ensurepip --upgrade
python -m pip --version
```

### Issue 3: TensorFlow Installation Failed

**Error:** Failed to install tensorflow

**Solutions:**

**Option A:** Try older version
```bash
pip install tensorflow==2.12.0
```

**Option B:** For Apple Silicon (M1/M2):
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

**Option C:** For CPU-only:
```bash
pip install tensorflow-cpu
```

### Issue 4: OpenCV Import Error

**Error:** `ImportError: libGL.so.1`

**Solution (Linux):**
```bash
sudo apt-get install libgl1-mesa-glx
```

### Issue 5: streamlit-drawable-canvas Issues

**Error:** Canvas not displaying

**Solution:**
```bash
pip uninstall streamlit-drawable-canvas
pip install streamlit-drawable-canvas==0.9.3
```

### Issue 6: PDF Support Not Working

**Error:** PDF processing fails

**Solution:**

**Windows:**
1. Download poppler: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\poppler`
3. Add `C:\poppler\bin` to PATH

**macOS:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

## Platform-Specific Instructions

### Windows

1. **Install Python:**
   - Download from python.org
   - Check "Add Python to PATH" during installation
   - Install for all users (recommended)

2. **Open Command Prompt:**
   - Press Win + R
   - Type `cmd` and press Enter

3. **Navigate to Project:**
   ```cmd
   cd C:\path\to\project
   ```

4. **Follow installation steps above**

5. **Common Windows Issues:**
   - **Long path error:** Enable long path support in Windows settings
   - **Permission error:** Run CMD as Administrator
   - **Antivirus blocking:** Add Python to antivirus exceptions

### macOS

1. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python:**
   ```bash
   brew install python@3.9
   ```

3. **Open Terminal:**
   - Press Cmd + Space
   - Type "Terminal"
   - Press Enter

4. **Navigate to Project:**
   ```bash
   cd ~/Downloads/handwriting-recognition
   ```

5. **Follow installation steps above**

6. **Apple Silicon (M1/M2) Specific:**
   ```bash
   # Install Rosetta if needed
   softwareupdate --install-rosetta
   
   # Use specific TensorFlow build
   pip install tensorflow-macos
   pip install tensorflow-metal
   ```

### Linux (Ubuntu/Debian)

1. **Update Package List:**
   ```bash
   sudo apt-get update
   ```

2. **Install Python and pip:**
   ```bash
   sudo apt-get install python3 python3-pip python3-venv
   ```

3. **Install System Dependencies:**
   ```bash
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

4. **For PDF Support:**
   ```bash
   sudo apt-get install poppler-utils
   ```

5. **Follow installation steps above**

### Linux (Fedora/RedHat)

1. **Install Python:**
   ```bash
   sudo dnf install python3 python3-pip
   ```

2. **Install System Dependencies:**
   ```bash
   sudo dnf install mesa-libGL glib2
   ```

3. **For PDF Support:**
   ```bash
   sudo dnf install poppler-utils
   ```

## GPU Support (Optional)

For faster training with NVIDIA GPU:

### Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5+
- NVIDIA GPU drivers
- CUDA Toolkit 11.2+
- cuDNN 8.1+

### Installation

1. **Install CUDA Toolkit:**
   - Download from NVIDIA website
   - Follow platform-specific instructions

2. **Install cuDNN:**
   - Download from NVIDIA Developer website
   - Extract and copy files to CUDA directory

3. **Install TensorFlow GPU:**
   ```bash
   pip install tensorflow-gpu
   ```

4. **Verify GPU Support:**
   ```python
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

## Post-Installation

### 1. Train the Model
```bash
python handwriting_model.py
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the App
Open browser to: `http://localhost:8501`

## Updating

To update to a newer version:

```bash
# Pull latest changes (if using git)
git pull

# Update packages
pip install -r requirements.txt --upgrade

# Retrain model if needed
python handwriting_model.py
```

## Uninstallation

To completely remove the application:

1. **Deactivate virtual environment:**
   ```bash
   deactivate
   ```

2. **Delete project folder:**
   - Delete the entire project directory
   - Or keep it and just delete the `venv` folder

3. **Remove from PATH** (if added)

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review README.md for usage instructions
3. Check QUICKSTART.md for common tasks
4. Search for error messages online
5. Create an issue on GitHub (if applicable)

## Next Steps

After successful installation:

1. ✅ Read QUICKSTART.md for quick start
2. ✅ Train your first model
3. ✅ Run the application
4. ✅ Try drawing a digit
5. ✅ Explore all features

---

**Installation complete! Enjoy using the Handwriting Recognition System! 🎉**
