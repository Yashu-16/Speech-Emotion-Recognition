"""
Speech Emotion Recognition - Step 4: Deep Learning Models
Train CNN and Feedforward Neural Networks using TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# DATA PREPARATION FOR DEEP LEARNING
# ============================================================================

def prepare_data_for_dl(feature_file, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare data for deep learning with train/validation/test splits
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR DEEP LEARNING")
    print("="*60)
    
    # Load features
    df = pd.read_csv(feature_file)
    print(f"\nLoaded {len(df)} samples")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['emotion', 'file_path']]
    X = df[feature_cols].values
    y = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # One-hot encode labels for neural networks
    y_categorical = to_categorical(y_encoded, num_classes)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_categorical,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Validation set: {X_val_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            scaler, label_encoder, num_classes)

# ============================================================================
# FEEDFORWARD NEURAL NETWORK (FULLY CONNECTED)
# ============================================================================

def create_feedforward_model(input_dim, num_classes, dropout_rate=0.5):
    """
    Create a Feedforward Neural Network (Multi-layer Perceptron)
    
    Architecture:
    - Input layer
    - Dense layers with ReLU activation
    - Dropout for regularization
    - Output layer with softmax
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second hidden layer
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Third hidden layer
        layers.Dense(64, activation='relu', name='dense_3'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Fourth hidden layer
        layers.Dense(32, activation='relu', name='dense_4'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

def train_feedforward_model(X_train, y_train, X_val, y_val, num_classes, epochs=100):
    """
    Train Feedforward Neural Network
    """
    print("\n" + "="*60)
    print("TRAINING FEEDFORWARD NEURAL NETWORK")
    print("="*60)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_feedforward_model(input_dim, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_feedforward_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    return model, history

# ============================================================================
# CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================================

def create_cnn_model(input_dim, num_classes, dropout_rate=0.5):
    """
    Create a 1D CNN for feature-based emotion recognition
    
    Architecture:
    - Reshape input for 1D convolution
    - Conv1D layers with pooling
    - Flatten and dense layers
    - Output layer
    """
    model = models.Sequential([
        # Reshape for CNN: (batch, features, 1)
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        
        # First Conv block
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Second Conv block
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Third Conv block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_cnn_model(X_train, y_train, X_val, y_val, num_classes, epochs=100):
    """
    Train 1D CNN model
    """
    print("\n" + "="*60)
    print("TRAINING CONVOLUTIONAL NEURAL NETWORK (1D CNN)")
    print("="*60)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_cnn_model(input_dim, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_cnn_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    return model, history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_dl_model(model, X_test, y_test, label_encoder, model_name):
    """
    Evaluate deep learning model on test set
    """
    print("\n" + "="*60)
    print(f"EVALUATING {model_name.upper()}")
    print("="*60)
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("SPEECH EMOTION RECOGNITION - DEEP LEARNING MODELS")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Step 1: Prepare data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, label_encoder, num_classes) = prepare_data_for_dl(
        'ravdess_features.csv',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Step 2: Train Feedforward Neural Network
    ffnn_model, ffnn_history = train_feedforward_model(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=100
    )
    
    # Plot training history
    plot_training_history(ffnn_history, "Feedforward NN")
    
    # Evaluate model
    ffnn_results = evaluate_dl_model(
        ffnn_model, X_test, y_test,
        label_encoder, "Feedforward NN"
    )
    
    # Step 3: Train CNN
    cnn_model, cnn_history = train_cnn_model(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=100
    )
    
    # Plot training history
    plot_training_history(cnn_history, "CNN")
    
    # Evaluate model
    cnn_results = evaluate_dl_model(
        cnn_model, X_test, y_test,
        label_encoder, "CNN"
    )
    
    # Step 4: Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nFeedforward NN Test Accuracy: {ffnn_results['accuracy']:.4f}")
    print(f"CNN Test Accuracy: {cnn_results['accuracy']:.4f}")
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    ffnn_model.save('feedforward_model.keras')
    cnn_model.save('cnn_model.keras')
    print("✓ Models saved successfully!")
    
    print("\n" + "="*60)
    print("DEEP LEARNING TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - feedforward_model.keras")
    print("  - cnn_model.keras")
    print("  - best_feedforward_model.keras (best checkpoint)")
    print("  - best_cnn_model.keras (best checkpoint)")
    print("  - Training history plots")
    print("  - Confusion matrices")
    print("\nNext step: Model fusion and advanced analysis (Step 5)")