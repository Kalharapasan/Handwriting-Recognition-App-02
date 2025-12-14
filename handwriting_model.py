"""
Advanced Handwriting Recognition Model Development
Complete model training and evaluation script
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Advanced Handwriting Recognition Model Training")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("=" * 60)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Data loading and preprocessing
print("\n[1/7] Loading MNIST Dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train_cat.shape}")
print(f"Test labels shape: {y_test_cat.shape}")

# Enhanced CNN Model
def create_enhanced_model():
    """Create an enhanced CNN model with batch normalization and dropout"""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def create_simple_model():
    """Create a simpler, faster model for quick training"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Model selection
print("\n[2/7] Creating Model Architecture...")
print("Choose model type:")
print("1. Enhanced Model (slower but more accurate)")
print("2. Simple Model (faster training)")

model_choice = input("Enter choice (1 or 2, press Enter for Simple): ").strip()

if model_choice == "1":
    print("Creating Enhanced Model...")
    model = create_enhanced_model()
    epochs = 50
else:
    print("Creating Simple Model...")
    model = create_simple_model()
    epochs = 20

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks
print("\n[3/7] Setting up Training Callbacks...")
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=10, 
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, 
        patience=5,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'models/best_handwriting_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir=f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        histogram_freq=1
    )
]

# Data augmentation
print("\n[4/7] Setting up Data Augmentation...")
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

# Train model
print("\n[5/7] Training Model...")
print(f"Training for up to {epochs} epochs...")
print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=128),
    epochs=epochs,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Evaluate model
print("\n[6/7] Evaluating Model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n{'='*60}")
print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"{'='*60}")

# Save final model
print("\n[7/7] Saving Models...")
model.save('models/handwriting_model.h5')
print("✓ Model saved to: models/handwriting_model.h5")

# Also save in Keras format
model.save('models/handwriting_model.keras')
print("✓ Model saved to: models/handwriting_model.keras")

# Plot training history
def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved to: plots/training_history.png")
    plt.show()

print("\nGenerating Training Visualizations...")
plot_training_history(history)

# Generate predictions for confusion matrix
print("\nGenerating Predictions for Analysis...")
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Digit Recognition', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved to: plots/confusion_matrix.png")
plt.show()

# Classification Report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)]))

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-"*60)
class_accuracy = []
for i in range(10):
    mask = y_test == i
    acc = np.mean(y_pred_classes[mask] == y_test[mask])
    class_accuracy.append(acc)
    print(f"Digit {i}: {acc:.4f} ({acc*100:.2f}%)")

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
bars = plt.bar(range(10), class_accuracy, color='steelblue', alpha=0.8)
plt.xlabel('Digit', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.ylim([0.9, 1.0])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('plots/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("\n✓ Per-class accuracy plot saved to: plots/per_class_accuracy.png")
plt.show()

# Sample predictions visualization
print("\nGenerating Sample Predictions Visualization...")
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < 20:
        # Get a random test sample
        idx = np.random.randint(0, len(x_test))
        
        # Display image
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        # Get prediction
        pred = np.argmax(y_pred[idx])
        true = y_test[idx]
        
        # Color code: green if correct, red if wrong
        color = 'green' if pred == true else 'red'
        ax.set_title(f'Pred: {pred}\nTrue: {true}', color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('plots/sample_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Sample predictions saved to: plots/sample_predictions.png")
plt.show()

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('logs/training_history.csv', index=False)
print("\n✓ Training history saved to: logs/training_history.csv")

# Save model configuration
print("\nSaving Model Configuration...")
config = {
    'model_type': 'Enhanced' if model_choice == "1" else 'Simple',
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'total_epochs': len(history.history['accuracy']),
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'total_parameters': model.count_params()
}

import json
with open('models/model_config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("✓ Model configuration saved to: models/model_config.json")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nFinal Results:")
print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  - Test Loss: {test_loss:.4f}")
print(f"  - Total Parameters: {model.count_params():,}")
print(f"  - Training Epochs: {len(history.history['accuracy'])}")
print(f"\nSaved Files:")
print(f"  - models/handwriting_model.h5")
print(f"  - models/handwriting_model.keras")
print(f"  - models/best_handwriting_model.h5")
print(f"  - models/model_config.json")
print(f"  - plots/training_history.png")
print(f"  - plots/confusion_matrix.png")
print(f"  - plots/per_class_accuracy.png")
print(f"  - plots/sample_predictions.png")
print(f"  - logs/training_history.csv")
print("="*60)
print("\nYou can now use this model with the Streamlit app!")
print("Run: streamlit run app.py")
print("="*60)
