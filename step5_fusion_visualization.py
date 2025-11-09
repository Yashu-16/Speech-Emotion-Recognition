"""
Speech Emotion Recognition - Step 5: Model Fusion & Advanced Visualization
Ensemble models and create t-SNE visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================

def load_all_models():
    """
    Load all trained models (classical ML and deep learning)
    """
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)
    
    models = {}
    
    # Load classical ML models
    try:
        models['SVM'] = joblib.load('svm_model.pkl')
        print("✓ Loaded SVM model")
    except:
        print("✗ SVM model not found")
    
    try:
        models['Random Forest'] = joblib.load('random_forest_model.pkl')
        print("✓ Loaded Random Forest model")
    except:
        print("✗ Random Forest model not found")
    
    # Load deep learning models
    try:
        models['Feedforward NN'] = keras.models.load_model('best_feedforward_model.keras')
        print("✓ Loaded Feedforward NN model")
    except:
        print("✗ Feedforward NN model not found")
    
    try:
        models['CNN'] = keras.models.load_model('best_cnn_model.keras')
        print("✓ Loaded CNN model")
    except:
        print("✗ CNN model not found")
    
    # Load preprocessing objects
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        print("✓ Loaded preprocessing objects")
    except:
        print("✗ Preprocessing objects not found")
        scaler, label_encoder = None, None
    
    return models, scaler, label_encoder

# ============================================================================
# ENSEMBLE FUSION METHODS
# ============================================================================

def voting_ensemble(models, X_test, y_test, label_encoder, voting='hard'):
    """
    Ensemble using voting (hard or soft)
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: True labels (one-hot encoded for DL models)
        label_encoder: Label encoder object
        voting: 'hard' for majority vote, 'soft' for probability averaging
    
    Returns:
        Predictions and accuracy
    """
    print("\n" + "="*60)
    print(f"ENSEMBLE: {voting.upper()} VOTING")
    print("="*60)
    
    predictions_list = []
    
    # Get predictions from each model
    for model_name, model in models.items():
        print(f"Getting predictions from {model_name}...")
        
        if model_name in ['Feedforward NN', 'CNN']:
            # Deep learning model - returns probabilities
            y_pred_proba = model.predict(X_test, verbose=0)
            
            if voting == 'hard':
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = y_pred_proba
        else:
            # Classical ML model
            if voting == 'hard':
                y_pred = model.predict(X_test)
            else:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_test)
                else:
                    # If no probability method, use hard prediction
                    y_pred = model.predict(X_test)
        
        predictions_list.append(y_pred)
    
    # Combine predictions
    if voting == 'hard':
        # Majority voting
        predictions_array = np.array(predictions_list)
        final_predictions = []
        
        for i in range(predictions_array.shape[1]):
            votes = predictions_array[:, i]
            final_pred = np.bincount(votes.astype(int)).argmax()
            final_predictions.append(final_pred)
        
        final_predictions = np.array(final_predictions)
    else:
        # Soft voting (average probabilities)
        # Ensure all predictions are probability arrays
        prob_arrays = []
        for pred in predictions_list:
            if len(pred.shape) == 1:
                # Convert hard predictions to one-hot
                num_classes = len(label_encoder.classes_)
                one_hot = np.zeros((len(pred), num_classes))
                one_hot[np.arange(len(pred)), pred.astype(int)] = 1
                prob_arrays.append(one_hot)
            else:
                prob_arrays.append(pred)
        
        # Average probabilities
        avg_probabilities = np.mean(prob_arrays, axis=0)
        final_predictions = np.argmax(avg_probabilities, axis=1)
    
    # Get true labels
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, final_predictions)
    
    print(f"\n{voting.capitalize()} Voting Ensemble Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, final_predictions,
        target_names=label_encoder.classes_
    ))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, final_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {voting.capitalize()} Voting Ensemble', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{voting}_voting_ensemble.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return final_predictions, accuracy

def weighted_ensemble(models, X_test, y_test, label_encoder, weights=None):
    """
    Weighted ensemble where each model has different weight
    """
    print("\n" + "="*60)
    print("ENSEMBLE: WEIGHTED VOTING")
    print("="*60)
    
    if weights is None:
        # Equal weights if not specified
        weights = {name: 1.0 for name in models.keys()}
    
    print(f"Weights: {weights}")
    
    weighted_probs = None
    total_weight = sum(weights.values())
    
    # Get weighted predictions
    for model_name, model in models.items():
        print(f"Getting predictions from {model_name}...")
        weight = weights[model_name]
        
        if model_name in ['Feedforward NN', 'CNN']:
            y_pred_proba = model.predict(X_test, verbose=0)
        else:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                # Convert hard predictions to probabilities
                y_pred = model.predict(X_test)
                num_classes = len(label_encoder.classes_)
                y_pred_proba = np.zeros((len(y_pred), num_classes))
                y_pred_proba[np.arange(len(y_pred)), y_pred] = 1
        
        # Add weighted probabilities
        if weighted_probs is None:
            weighted_probs = weight * y_pred_proba
        else:
            weighted_probs += weight * y_pred_proba
    
    # Normalize by total weight
    weighted_probs /= total_weight
    
    # Get final predictions
    final_predictions = np.argmax(weighted_probs, axis=1)
    
    # Get true labels
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, final_predictions)
    
    print(f"\nWeighted Ensemble Accuracy: {accuracy:.4f}")
    
    return final_predictions, accuracy

# ============================================================================
# t-SNE VISUALIZATION
# ============================================================================

def visualize_tsne(X, y, label_encoder, title="t-SNE Visualization", perplexity=30):
    """
    Create t-SNE visualization of feature space
    """
    print("\n" + "="*60)
    print("CREATING t-SNE VISUALIZATION")
    print("="*60)
    print("This may take a few minutes...")
    
    # Convert one-hot encoded labels to integers if needed
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    # Reduce dimensionality with PCA first (speeds up t-SNE)
    if X.shape[1] > 50:
        print("\nApplying PCA preprocessing...")
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        X_pca = X
    
    # Apply t-SNE
    print("\nApplying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each emotion class
    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=[colors[idx]], label=label_encoder.classes_[label],
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ t-SNE visualization saved!")

# ============================================================================
# PCA VISUALIZATION
# ============================================================================

def visualize_pca(X, y, label_encoder, title="PCA Visualization"):
    """
    Create PCA visualization showing first 2 principal components
    """
    print("\n" + "="*60)
    print("CREATING PCA VISUALIZATION")
    print("="*60)
    
    # Convert one-hot encoded labels to integers if needed
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each emotion class
    for idx, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[idx]], label=label_encoder.classes_[label],
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ PCA visualization saved!")

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def create_final_comparison(results_dict):
    """
    Create comprehensive comparison of all models including ensembles
    """
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for model_name, accuracy in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': accuracy
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
    
    print("\n", comparison_df.to_string(index=False))
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(comparison_df['Model'], comparison_df['Test Accuracy'], 
                   color='skyblue', edgecolor='black', linewidth=1.5)
    
    # Color the best model
    max_idx = comparison_df['Test Accuracy'].argmax()
    bars[max_idx].set_color('gold')
    bars[max_idx].set_edgecolor('darkgoldenrod')
    bars[max_idx].set_linewidth(2)
    
    plt.title('Final Model Comparison - Test Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim([0, 1.0])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return comparison_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("SPEECH EMOTION RECOGNITION - MODEL FUSION & VISUALIZATION")
    print("="*60)
    
    # Load models
    models, scaler, label_encoder = load_all_models()
    
    if len(models) == 0:
        print("\n❌ No models found! Please train models first (Steps 3-4).")
        exit()
    
    # Load test data
    print("\nLoading test data...")
    features_df = pd.read_csv('ravdess_features.csv')
    feature_cols = [col for col in features_df.columns if col not in ['emotion', 'file_path']]
    X = features_df[feature_cols].values
    y = features_df['emotion'].values
    
    # Encode and split
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    y_encoded = label_encoder.transform(y)
    y_categorical = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Store all results
    all_results = {}
    
    # Individual model accuracies (if available from previous steps)
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    for model_name, model in models.items():
        if model_name in ['Feedforward NN', 'CNN']:
            y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        else:
            y_pred = model.predict(X_test_scaled)
        
        y_true = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        all_results[model_name] = accuracy
        print(f"{model_name}: {accuracy:.4f}")
    
    # Ensemble methods
    if len(models) >= 2:
        # Hard voting
        _, hard_voting_acc = voting_ensemble(
            models, X_test_scaled, y_test, label_encoder, voting='hard'
        )
        all_results['Hard Voting Ensemble'] = hard_voting_acc
        
        # Soft voting
        _, soft_voting_acc = voting_ensemble(
            models, X_test_scaled, y_test, label_encoder, voting='soft'
        )
        all_results['Soft Voting Ensemble'] = soft_voting_acc
    
    # Create visualizations
    visualize_pca(X_test_scaled, y_test, label_encoder, "PCA - Test Set")
    visualize_tsne(X_test_scaled, y_test, label_encoder, "t-SNE - Test Set", perplexity=30)
    
    # Final comparison
    comparison_df = create_final_comparison(all_results)
    
    # Save results
    comparison_df.to_csv('final_results.csv', index=False)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nAll generated files:")
    print("  - final_results.csv")
    print("  - final_model_comparison.png")
    print("  - pca_visualization.png")
    print("  - tsne_visualization.png")
    print("  - confusion_matrix_*_voting_ensemble.png")
    print("\nBest performing model:")
    best_model = comparison_df.iloc[0]
    print(f"  {best_model['Model']}: {best_model['Test Accuracy']:.4f}")