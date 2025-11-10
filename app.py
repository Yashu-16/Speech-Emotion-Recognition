"""
Streamlit Web Application for Speech Emotion Recognition
TEMPORARY VERSION - Without TensorFlow (Classical ML only)
Run with: streamlit run app_no_tf.py
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎭 Speech Emotion Recognition")
st.markdown("### Detect emotions from speech using Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["SVM", "Random Forest", "Ensemble (SVM + RF)"]
    )
    
    st.markdown("---")
    st.header("📊 About")
    st.info("""
    This application recognizes emotions from speech audio:
    - **Happy** 😊
    - **Sad** 😢
    - **Angry** 😠
    - **Neutral** 😐
    - **Fearful** 😨
    - **Surprised** 😲
    - **Disgusted** 🤢
    - **Calm** 😌
    
    **Note**: Using Classical ML models only.
    Install TensorFlow to enable Deep Learning models.
    """)
    
    st.markdown("---")
    st.markdown("**Team:** Juhi Dixit & Yash Randhe")

# Feature extraction functions
@st.cache_data
def extract_features(audio, sr):
    """Extract all acoustic features from audio"""
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    mfcc_features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) > 0:
        pitch_features = np.array([
            np.mean(pitch_values),
            np.std(pitch_values),
            np.max(pitch_values),
            np.min(pitch_values)
        ])
    else:
        pitch_features = np.zeros(4)
    
    # Energy
    rms = librosa.feature.rms(y=audio)[0]
    energy_features = np.array([
        np.mean(rms),
        np.std(rms),
        np.max(rms),
        np.min(rms)
    ])
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_features = np.array([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff)
    ])
    
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_features = np.array([
        np.mean(zcr),
        np.std(zcr),
        np.max(zcr),
        np.min(zcr)
    ])
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_features = np.array([
        np.mean(chroma),
        np.std(chroma),
        np.max(chroma),
        np.min(chroma)
    ])
    
    # Combine all features
    all_features = np.concatenate([
        mfcc_features,
        pitch_features,
        energy_features,
        spectral_features,
        zcr_features,
        chroma_features
    ])
    
    return all_features

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        models['SVM'] = joblib.load('svm_model.pkl')
        st.sidebar.success("✓ SVM loaded")
    except Exception as e:
        st.sidebar.error(f"✗ SVM not found: {e}")
    
    try:
        models['Random Forest'] = joblib.load('random_forest_model.pkl')
        st.sidebar.success("✓ Random Forest loaded")
    except Exception as e:
        st.sidebar.error(f"✗ Random Forest not found: {e}")
    
    try:
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        st.sidebar.success("✓ Preprocessing loaded")
    except Exception as e:
        st.sidebar.error(f"✗ Preprocessing not found: {e}")
    
    return models

# Predict emotion
def predict_emotion(audio, sr, models, model_choice):
    """Predict emotion from audio"""
    
    # Extract features
    features = extract_features(audio, sr)
    
    # Scale features
    if 'scaler' not in models:
        st.error("Scaler not found!")
        return None, None
    
    features_scaled = models['scaler'].transform(features.reshape(1, -1))
    
    # Get prediction based on model choice
    if model_choice == "Ensemble (SVM + RF)":
        predictions = []
        
        if 'SVM' in models:
            if hasattr(models['SVM'], 'predict_proba'):
                pred = models['SVM'].predict_proba(features_scaled)[0]
            else:
                pred_class = models['SVM'].predict(features_scaled)[0]
                pred = np.zeros(len(models['label_encoder'].classes_))
                pred[pred_class] = 1.0
            predictions.append(pred)
        
        if 'Random Forest' in models:
            if hasattr(models['Random Forest'], 'predict_proba'):
                pred = models['Random Forest'].predict_proba(features_scaled)[0]
            else:
                pred_class = models['Random Forest'].predict(features_scaled)[0]
                pred = np.zeros(len(models['label_encoder'].classes_))
                pred[pred_class] = 1.0
            predictions.append(pred)
        
        if len(predictions) > 0:
            avg_pred = np.mean(predictions, axis=0)
            emotion_idx = np.argmax(avg_pred)
            confidence = avg_pred
        else:
            st.error("No models available!")
            return None, None
    
    elif model_choice == "SVM":
        if 'SVM' not in models:
            st.error("SVM model not found!")
            return None, None
        emotion_idx = models['SVM'].predict(features_scaled)[0]
        # Check if SVM has predict_proba
        if hasattr(models['SVM'], 'predict_proba'):
            confidence = models['SVM'].predict_proba(features_scaled)[0]
        else:
            # Create one-hot encoded confidence if no probabilities
            num_classes = len(models['label_encoder'].classes_)
            confidence = np.zeros(num_classes)
            confidence[emotion_idx] = 1.0
    
    elif model_choice == "Random Forest":
        if 'Random Forest' not in models:
            st.error("Random Forest model not found!")
            return None, None
        emotion_idx = models['Random Forest'].predict(features_scaled)[0]
        if hasattr(models['Random Forest'], 'predict_proba'):
            confidence = models['Random Forest'].predict_proba(features_scaled)[0]
        else:
            num_classes = len(models['label_encoder'].classes_)
            confidence = np.zeros(num_classes)
            confidence[emotion_idx] = 1.0
    
    # Get emotion label
    if 'label_encoder' in models:
        emotion = models['label_encoder'].classes_[emotion_idx]
    else:
        emotion = f"Emotion {emotion_idx}"
    
    return emotion, confidence

# Visualize audio
def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_spectrogram(audio, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_confidence(confidence, labels):
    """Plot confidence scores"""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, confidence, color=colors)
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_title('Emotion Confidence Scores', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, confidence)):
        ax.text(val, i, f' {val:.2%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main application
def main():
    # Load models
    models = load_models()
    
    if not models or 'scaler' not in models:
        st.error("❌ Models not loaded! Please train models first (steps 1-3).")
        st.info("💡 Make sure these files are in the same directory:\n- svm_model.pkl\n- random_forest_model.pkl\n- scaler.pkl\n- label_encoder.pkl")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📁 Upload Audio", "📊 About"])
    
    with tab1:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3)",
            type=['wav', 'mp3', 'ogg', 'flac']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            audio, sr = librosa.load(tmp_path, sr=22050)
            
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Predict button
            if st.button("🎯 Predict Emotion", type="primary"):
                with st.spinner("Analyzing audio..."):
                    emotion, confidence = predict_emotion(audio, sr, models, model_choice)
                
                if emotion is not None:
                    # Display result
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        emotion_icons = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠',
                            'neutral': '😐', 'fearful': '😨', 'surprised': '😲',
                            'disgust': '🤢', 'calm': '😌'
                        }
                        icon = emotion_icons.get(emotion.lower(), '🎭')
                        st.success(f"## {icon} {emotion.upper()}")
                        st.metric("Confidence", f"{confidence[np.argmax(confidence)]:.1%}")
                    
                    with col2:
                        # Confidence plot
                        fig = plot_confidence(confidence, models['label_encoder'].classes_)
                        st.pyplot(fig)
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📊 Audio Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = plot_waveform(audio, sr)
                        st.pyplot(fig)
                    
                    with col2:
                        fig = plot_spectrogram(audio, sr)
                        st.pyplot(fig)
            
            # Clean up temp file
            os.unlink(tmp_path)
    
    with tab2:
        st.header("About This Project")
        st.markdown("""
        ### 🎯 Speech Emotion Recognition System
        
        This application uses **Classical Machine Learning** to detect emotions from speech.
        
        #### 📊 Models Used:
        - **SVM (Support Vector Machine)**: 60-75% accuracy
        - **Random Forest**: 65-80% accuracy
        - **Ensemble**: Combines both models for better performance
        
        #### 🔬 Features Extracted:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Pitch and frequency
        - Energy and intensity
        - Spectral features
        - Zero-crossing rate
        - Chroma features
        
        #### 👥 Team:
        - Juhi Dixit
        - Yash Randhe
        
        #### 📚 Course:
        CPE646 Pattern Recognition
        
        ---
        
        **Note**: Deep Learning models (CNN, Feedforward NN) require TensorFlow.
        To enable them, install TensorFlow properly and use the full version of the app.
        """)

if __name__ == "__main__":
    main()