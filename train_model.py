"""
Advanced Handwriting Recognition Model Training
Enhanced version based on Jupyter Notebook with additional features
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("="*70)
print("        HANDWRITING RECOGNITION MODEL TRAINING")
print("="*70)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# =============================================================================
# STEP 1: Load and Preprocess Data
# =============================================================================
print("\n[STEP 1/8] Loading MNIST Dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
print("[STEP 2/8] Preprocessing Data...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimension (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"✓ Training data shape: {x_train.shape}")
print(f"✓ Test data shape: {x_test.shape}")
print(f"✓ Training labels shape: {y_train_cat.shape}")
print(f"✓ Test labels shape: {y_test_cat.shape}")
print(f"✓ Number of classes: 10 (digits 0-9)")

# Visualize sample images
print("\n[INFO] Visualizing sample training images...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Training Images', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('plots/sample_training_images.png', dpi=150, bbox_inches='tight')
print(f"✓ Sample images saved to: plots/sample_training_images.png")
plt.close()

# =============================================================================
# STEP 2: Create Enhanced CNN Model
# =============================================================================
print("\n[STEP 3/8] Building Enhanced CNN Model...")

def create_enhanced_model():
    """
    Create an enhanced CNN model with:
    - Multiple convolutional blocks
    - Batch normalization for faster training
    - Dropout for regularization
    - Dense layers for classification
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create model
model = create_enhanced_model()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
model.summary()
print("="*70)

# Calculate and display model size
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Estimated Model Size: ~{total_params * 4 / (1024*1024):.2f} MB")

# =============================================================================
# STEP 3: Setup Training Configuration
# =============================================================================
print("\n[STEP 4/8] Configuring Training Parameters...")

# Callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=10, 
        restore_best_weights=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, 
        patience=5,
        monitor='val_loss',
        verbose=1,
        min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        'models/best_handwriting_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        'logs/training_log.csv',
        append=True
    )
]

print("✓ Callbacks configured:")
print("  - Early Stopping (patience=10)")
print("  - Learning Rate Reduction (factor=0.5, patience=5)")
print("  - Model Checkpoint (save best model)")
print("  - CSV Logger (save metrics)")

# Data augmentation for better generalization
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,        # Random rotation ±10 degrees
    width_shift_range=0.1,    # Horizontal shift ±10%
    height_shift_range=0.1,   # Vertical shift ±10%
    zoom_range=0.1            # Random zoom ±10%
)

print("\n✓ Data Augmentation configured:")
print("  - Rotation: ±10 degrees")
print("  - Width shift: ±10%")
print("  - Height shift: ±10%")
print("  - Zoom: ±10%")

# =============================================================================
# STEP 4: Train the Model
# =============================================================================
print("\n[STEP 5/8] Training Model...")
print("="*70)
print(f"Training Configuration:")
print(f"  - Epochs: 100 (with early stopping)")
print(f"  - Batch Size: 128")
print(f"  - Optimizer: Adam (lr=0.001)")
print(f"  - Training samples: {len(x_train):,}")
print(f"  - Validation samples: {len(x_test):,}")
print("="*70)
print("\nTraining in progress... This may take 20-40 minutes.")
print("Press Ctrl+C to stop training early (best model will be restored)\n")

try:
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=128),
        epochs=100,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    training_completed = True
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user!")
    training_completed = False

# =============================================================================
# STEP 5: Evaluate the Model
# =============================================================================
print("\n[STEP 6/8] Evaluating Model Performance...")
print("="*70)

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"FINAL TEST RESULTS:")
print(f"  • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Test Loss: {test_loss:.4f}")
print("="*70)

# =============================================================================
# STEP 6: Save the Model
# =============================================================================
print("\n[STEP 7/8] Saving Model and Configuration...")

# Save in HDF5 format
model.save('models/handwriting_model.h5')
print("✓ Model saved: models/handwriting_model.h5")

# Save in Keras format (newer format)
try:
    model.save('models/handwriting_model.keras')
    print("✓ Model saved: models/handwriting_model.keras")
except:
    print("⚠️  Could not save .keras format (requires TF 2.12+)")

# Save model configuration
config = {
    'model_type': 'Enhanced CNN',
    'architecture': 'Conv-BN-Conv-BN-Pool-Dropout × 3 + Dense',
    'input_shape': [28, 28, 1],
    'output_classes': 10,
    'total_parameters': int(total_params),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'training_completed': training_completed,
    'total_epochs': len(history.history['accuracy']) if training_completed else 0,
    'final_train_accuracy': float(history.history['accuracy'][-1]) if training_completed else 0,
    'final_val_accuracy': float(history.history['val_accuracy'][-1]) if training_completed else 0,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'tensorflow_version': tf.__version__
}

with open('models/model_config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("✓ Configuration saved: models/model_config.json")

# =============================================================================
# STEP 7: Generate Visualizations and Reports
# =============================================================================
print("\n[STEP 8/8] Generating Visualizations and Reports...")

if training_completed:
    # Plot training history
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training history plot: plots/training_history.png")
    
    plot_training_history(history)
    
    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('logs/training_history.csv', index=False)
    print("✓ Training history CSV: logs/training_history.csv")

# Generate predictions
print("\nGenerating predictions on test set...")
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Handwriting Recognition', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Digit', fontsize=12)
plt.ylabel('True Digit', fontsize=12)
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix: plots/confusion_matrix.png")

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(y_test, y_pred_classes, 
                               target_names=[str(i) for i in range(10)],
                               digits=4)
print(report)

# Save classification report
with open('logs/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(report)
print("\n✓ Classification report: logs/classification_report.txt")

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS ACCURACY")
print("="*70)
class_accuracy = []
for i in range(10):
    mask = y_test == i
    acc = np.mean(y_pred_classes[mask] == y_test[mask])
    class_accuracy.append(acc)
    print(f"  Digit {i}: {acc:.4f} ({acc*100:.2f}%)")

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
bars = plt.bar(range(10), class_accuracy, color='steelblue', alpha=0.8, edgecolor='navy')
plt.xlabel('Digit', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.ylim([min(class_accuracy) - 0.01, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(range(10))

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Per-class accuracy plot: plots/per_class_accuracy.png")

# Sample predictions with images
print("\nGenerating sample predictions visualization...")
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Predictions from Test Set', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(x_test))
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    
    pred = np.argmax(y_pred[idx])
    true = y_test[idx]
    conf = np.max(y_pred[idx])
    
    color = 'green' if pred == true else 'red'
    ax.set_title(f'Pred: {pred} | True: {true}\nConf: {conf:.2%}', 
                color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('plots/sample_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Sample predictions: plots/sample_predictions.png")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*70)
print("                    TRAINING COMPLETED!")
print("="*70)
print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"  • Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Final Test Loss: {test_loss:.4f}")
if training_completed:
    print(f"  • Training Epochs: {len(history.history['accuracy'])}")
    print(f"  • Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"  • Total Parameters: {total_params:,}")
print(f"  • Model Size: ~{total_params * 4 / (1024*1024):.2f} MB")

print(f"\n💾 SAVED FILES:")
print(f"  Models:")
print(f"    • models/handwriting_model.h5")
print(f"    • models/best_handwriting_model.h5")
print(f"    • models/model_config.json")
print(f"  Visualizations:")
if training_completed:
    print(f"    • plots/training_history.png")
print(f"    • plots/confusion_matrix.png")
print(f"    • plots/per_class_accuracy.png")
print(f"    • plots/sample_predictions.png")
print(f"    • plots/sample_training_images.png")
print(f"  Logs:")
if training_completed:
    print(f"    • logs/training_history.csv")
print(f"    • logs/training_log.csv")
print(f"    • logs/classification_report.txt")

print(f"\n🚀 NEXT STEPS:")
print(f"  1. Run the Streamlit app:")
print(f"     streamlit run app.py")
print(f"  2. Test with your own handwritten digits")
print(f"  3. View analytics and performance metrics")

print("\n" + "="*70)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
print("\n✨ Model ready for deployment! ✨\n")
