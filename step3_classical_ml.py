"""
Speech Emotion Recognition - Step 3: Classical Machine Learning Models
Train SVM and Random Forest models for emotion classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, precision_score, recall_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(feature_file, test_size=0.2, random_state=42):
    """
    Load features and prepare train/test splits
    
    Args:
        feature_file: Path to CSV file with features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load features
    df = pd.read_csv(feature_file)
    print(f"\nLoaded {len(df)} samples from '{feature_file}'")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['emotion', 'file_path']]
    X = df[feature_cols].values
    y = df['emotion'].values
    
    print(f"Feature shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"\nEmotion distribution:")
    print(pd.Series(y).value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nEncoded labels: {label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Data preparation complete!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

# ============================================================================
# MODEL TRAINING: SUPPORT VECTOR MACHINE (SVM)
# ============================================================================

def train_svm(X_train, y_train, X_test, y_test, tune_hyperparameters=True):
    """
    Train Support Vector Machine classifier
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        tune_hyperparameters: Whether to perform grid search
    
    Returns:
        Trained SVM model
    """
    print("\n" + "="*60)
    print("TRAINING SVM MODEL")
    print("="*60)
    
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning (this may take a while)...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        # Grid search with cross-validation
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        best_svm = grid_search.best_estimator_
    else:
        print("\nTraining with default parameters...")
        best_svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        best_svm.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    return best_svm

# ============================================================================
# MODEL TRAINING: RANDOM FOREST
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, tune_hyperparameters=True):
    """
    Train Random Forest classifier
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        tune_hyperparameters: Whether to perform grid search
    
    Returns:
        Trained Random Forest model
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        best_rf = grid_search.best_estimator_
    else:
        print("\nTraining with default parameters...")
        best_rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            random_state=42
        )
        best_rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    return best_rf

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, label_encoder, model_name="Model"):
    """
    Comprehensive evaluation of a trained model
    """
    print("\n" + "="*60)
    print(f"EVALUATING {model_name.upper()}")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

def plot_confusion_matrix(cm, labels, model_name, save_path=None):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def compare_models(results_dict, label_encoder):
    """
    Compare multiple models side by side
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar chart of metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        values = [results['accuracy'], results['precision'], 
                 results['recall'], results['f1_score']]
        axes[0].bar(x + idx*width, values, width, label=model_name)
    
    axes[0].set_xlabel('Metrics', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width/2)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Per-class accuracy
    for model_name, results in results_dict.items():
        cm = results['confusion_matrix']
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        axes[1].plot(label_encoder.classes_, per_class_acc, 
                    marker='o', label=model_name, linewidth=2)
    
    axes[1].set_xlabel('Emotion', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Per-Emotion Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return comparison_df

# ============================================================================
# SAVE MODELS
# ============================================================================

def save_models(models_dict, scaler, label_encoder):
    """
    Save trained models and preprocessing objects
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    for model_name, model in models_dict.items():
        filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved {model_name} to '{filename}'")
    
    # Save scaler and label encoder
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print(f"✓ Saved scaler to 'scaler.pkl'")
    print(f"✓ Saved label encoder to 'label_encoder.pkl'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("SPEECH EMOTION RECOGNITION - CLASSICAL ML MODELS")
    print("="*60)
    
    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_prepare_data(
        'ravdess_features.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Step 2: Train SVM
    svm_model = train_svm(
        X_train, y_train, X_test, y_test,
        tune_hyperparameters=False  # Set to True for better results (takes longer)
    )
    
    # Step 3: Train Random Forest
    rf_model = train_random_forest(
        X_train, y_train, X_test, y_test,
        tune_hyperparameters=False  # Set to True for better results (takes longer)
    )
    
    # Step 4: Evaluate both models
    svm_results = evaluate_model(svm_model, X_test, y_test, label_encoder, "SVM")
    rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    
    # Step 5: Plot confusion matrices
    plot_confusion_matrix(
        svm_results['confusion_matrix'], 
        label_encoder.classes_,
        'SVM',
        save_path='confusion_matrix_svm.png'
    )
    
    plot_confusion_matrix(
        rf_results['confusion_matrix'],
        label_encoder.classes_,
        'Random Forest',
        save_path='confusion_matrix_rf.png'
    )
    
    # Step 6: Compare models
    results_dict = {
        'SVM': svm_results,
        'Random Forest': rf_results
    }
    comparison_df = compare_models(results_dict, label_encoder)
    
    # Step 7: Save models
    models_dict = {
        'SVM': svm_model,
        'Random Forest': rf_model
    }
    save_models(models_dict, scaler, label_encoder)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - svm_model.pkl")
    print("  - random_forest_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoder.pkl")
    print("  - confusion_matrix_svm.png")
    print("  - confusion_matrix_rf.png")
    print("  - model_comparison.png")
    print("\nNext step: Train deep learning models (Step 4)")