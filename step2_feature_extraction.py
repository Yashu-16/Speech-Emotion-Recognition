"""
Speech Emotion Recognition - Step 2: Feature Extraction
Extract MFCCs, pitch, energy, and other acoustic features from audio files.
"""

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_mfcc_features(audio, sr, n_mfcc=13):
    """
    Extract MFCC (Mel-frequency cepstral coefficients) features
    MFCCs capture the spectral envelope of the sound
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate statistics for each MFCC coefficient
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    
    return np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])

def extract_pitch_features(audio, sr):
    """
    Extract pitch-related features using fundamental frequency (F0)
    Pitch reflects the perceived frequency of sound
    """
    # Extract pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    
    # Get pitch values where magnitude is above threshold
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) == 0:
        return np.zeros(4)
    
    pitch_values = np.array(pitch_values)
    
    return np.array([
        np.mean(pitch_values),      # Mean pitch
        np.std(pitch_values),       # Pitch variation
        np.max(pitch_values),       # Maximum pitch
        np.min(pitch_values)        # Minimum pitch
    ])

def extract_energy_features(audio):
    """
    Extract energy-related features (RMS energy)
    Energy reflects the loudness/intensity of the signal
    """
    rms = librosa.feature.rms(y=audio)[0]
    
    return np.array([
        np.mean(rms),               # Mean energy
        np.std(rms),                # Energy variation
        np.max(rms),                # Maximum energy
        np.min(rms)                 # Minimum energy
    ])

def extract_spectral_features(audio, sr):
    """
    Extract spectral features: centroid, bandwidth, rolloff
    These capture the frequency distribution of the signal
    """
    # Spectral centroid: center of mass of the spectrum
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    # Spectral bandwidth: width of the spectrum
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Spectral rolloff: frequency below which a certain percentage of energy is contained
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    return np.array([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff)
    ])

def extract_zcr_features(audio):
    """
    Extract Zero Crossing Rate (ZCR) features
    ZCR is the rate at which the signal changes sign
    """
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    return np.array([
        np.mean(zcr),
        np.std(zcr),
        np.max(zcr),
        np.min(zcr)
    ])

def extract_chroma_features(audio, sr):
    """
    Extract chroma features (pitch class profiles)
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    return np.array([
        np.mean(chroma),
        np.std(chroma),
        np.max(chroma),
        np.min(chroma)
    ])

# ============================================================================
# COMPREHENSIVE FEATURE EXTRACTION
# ============================================================================

def extract_all_features(file_path, sr=22050):
    """
    Extract all acoustic features from an audio file
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
    
    Returns:
        Dictionary containing all features
    """
    try:
        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=sr)
        
        # Extract all features
        mfcc_features = extract_mfcc_features(audio, sample_rate)
        pitch_features = extract_pitch_features(audio, sample_rate)
        energy_features = extract_energy_features(audio)
        spectral_features = extract_spectral_features(audio, sample_rate)
        zcr_features = extract_zcr_features(audio)
        chroma_features = extract_chroma_features(audio, sample_rate)
        
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
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def create_feature_names():
    """
    Create descriptive names for all features
    """
    feature_names = []
    
    # MFCC features (13 coefficients × 4 statistics = 52 features)
    for i in range(13):
        feature_names.extend([
            f'mfcc_{i}_mean',
            f'mfcc_{i}_std',
            f'mfcc_{i}_max',
            f'mfcc_{i}_min'
        ])
    
    # Pitch features (4)
    feature_names.extend(['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min'])
    
    # Energy features (4)
    feature_names.extend(['energy_mean', 'energy_std', 'energy_max', 'energy_min'])
    
    # Spectral features (6)
    feature_names.extend([
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std'
    ])
    
    # ZCR features (4)
    feature_names.extend(['zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min'])
    
    # Chroma features (4)
    feature_names.extend(['chroma_mean', 'chroma_std', 'chroma_max', 'chroma_min'])
    
    return feature_names

# ============================================================================
# PROCESS ENTIRE DATASET
# ============================================================================

def process_dataset(catalog_df, output_file='features.csv', sr=22050):
    """
    Extract features from all files in the dataset catalog
    
    Args:
        catalog_df: DataFrame with file paths and labels
        output_file: Name of output CSV file
        sr: Sample rate
    
    Returns:
        DataFrame with features and labels
    """
    print(f"\nExtracting features from {len(catalog_df)} audio files...")
    print("This may take several minutes...\n")
    
    features_list = []
    labels_list = []
    file_paths = []
    
    # Process each file with progress bar
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df)):
        file_path = row['file_path']
        emotion = row['emotion']
        
        # Extract features
        features = extract_all_features(file_path, sr=sr)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(emotion)
            file_paths.append(file_path)
    
    # Create DataFrame
    feature_names = create_feature_names()
    features_df = pd.DataFrame(features_list, columns=feature_names)
    features_df['emotion'] = labels_list
    features_df['file_path'] = file_paths
    
    # Save to CSV
    features_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"✓ Saved {len(features_df)} samples to '{output_file}'")
    print(f"✓ Total features per sample: {len(feature_names)}")
    
    return features_df

# ============================================================================
# FEATURE ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_features(features_df):
    """
    Analyze and display feature statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Exclude non-feature columns
    feature_cols = [col for col in features_df.columns if col not in ['emotion', 'file_path']]
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total samples: {len(features_df)}")
    print(f"\nFeature value ranges:")
    print(features_df[feature_cols].describe())
    
    # Visualize feature distributions by emotion
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sample_features = ['mfcc_0_mean', 'pitch_mean', 'energy_mean', 'spectral_centroid_mean']
    
    for idx, feature in enumerate(sample_features):
        row = idx // 2
        col = idx % 2
        
        for emotion in features_df['emotion'].unique():
            emotion_data = features_df[features_df['emotion'] == emotion][feature]
            axes[row, col].hist(emotion_data, alpha=0.5, label=emotion, bins=20)
        
        axes[row, col].set_title(f'{feature} Distribution by Emotion', fontsize=12)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("SPEECH EMOTION RECOGNITION - FEATURE EXTRACTION")
    print("="*60)
    
    # Load the catalog created in Step 1
    try:
        # Load RAVDESS catalog
        ravdess_catalog = pd.read_csv('ravdess_catalog.csv')
        print(f"\nLoaded RAVDESS catalog: {len(ravdess_catalog)} files")
        
        # Extract features
        ravdess_features = process_dataset(
            ravdess_catalog, 
            output_file='ravdess_features.csv',
            sr=22050
        )
        
        # Analyze features
        analyze_features(ravdess_features)
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION COMPLETE!")
        print("="*60)
        print("\nOutput files:")
        print("  - ravdess_features.csv (feature dataset)")
        print("  - feature_distributions.png (visualization)")
        print("\nNext step: Train machine learning models (Step 3)")
        
    except FileNotFoundError:
        print("\n❌ Error: Could not find catalog file.")
        print("Please run Step 1 (Data Exploration) first to create the catalog.")
    
    # Optional: Process CREMA-D dataset
    try:
        crema_catalog = pd.read_csv('crema_catalog.csv')
        print(f"\n\nProcessing CREMA-D dataset: {len(crema_catalog)} files")
        crema_features = process_dataset(
            crema_catalog,
            output_file='crema_features.csv',
            sr=22050
        )
    except FileNotFoundError:
        print("\nCREMA-D catalog not found. Skipping...")