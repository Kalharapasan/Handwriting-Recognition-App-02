#!/usr/bin/env python3
import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, total, text):
    print(f"\n[{step}/{total}] {text}")

def check_python_version():
    print_step(1, 6, "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ is required. You have Python {version.major}.{version.minor}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    print_step(2, 6, "Creating project directories...")
    
    directories = [
        'models',
        'uploaded_files/images',
        'uploaded_files/documents',
        'uploaded_files/drawings',
        'logs',
        'plots',
        'exports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return True

def install_requirements():
    print_step(3, 6, "Installing Python packages...")
    print("This may take a few minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            "requirements.txt",
            "--upgrade"
        ])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        print("Try running manually: pip install -r requirements.txt")
        return False

def check_optional_dependencies():
    print_step(4, 6, "Checking optional dependencies...")
    
    # Check for poppler (PDF support)
    system = platform.system()
    
    if system == "Linux":
        print("For PDF support on Linux, install: sudo apt-get install poppler-utils")
    elif system == "Darwin":  # macOS
        print("For PDF support on macOS, install: brew install poppler")
    elif system == "Windows":
        print("For PDF support on Windows, download from:")
        print("https://github.com/oschwartz10612/poppler-windows/releases")
    
    return True

def test_imports():
    print_step(5, 6, "Testing package imports...")
    
    required_packages = [
        ('tensorflow', 'TensorFlow'),
        ('streamlit', 'Streamlit'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn')
    ]
    
    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - Failed to import")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Some packages failed to import: {', '.join(failed)}")
        return False
    
    return True

def check_model_exists():
    print_step(6, 6, "Checking for trained model...")
    
    model_paths = [
        'models/handwriting_model.h5',
        'models/handwriting_model.keras',
        'models/best_handwriting_model.h5'
    ]
    
    model_found = any(os.path.exists(path) for path in model_paths)
    
    if model_found:
        print("✓ Trained model found")
    else:
        print("⚠️  No trained model found")
        print("\nTo train a model, run:")
        print("  python handwriting_model.py")
    
    return True

def main():
    print_header("Handwriting Recognition System - Setup")
    print("This script will set up your environment")
    
    # Run all setup steps
    steps = [
        check_python_version,
        create_directories,
        install_requirements,
        check_optional_dependencies,
        test_imports,
        check_model_exists
    ]
    
    success = True
    for step in steps:
        if not step():
            success = False
            break
    
    # Print summary
    print("\n" + "="*60)
    if success:
        print("  ✅ SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext Steps:")
        print("  1. Train a model (if not already done):")
        print("     python handwriting_model.py")
        print("\n  2. Start the application:")
        print("     streamlit run app.py")
        print("\n  3. Open browser to: http://localhost:8501")
    else:
        print("  ⚠️  SETUP COMPLETED WITH WARNINGS")
        print("="*60)
        print("\nPlease address the issues above and try again.")
        print("For help, see: README.md")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
