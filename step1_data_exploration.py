"""
Speech Emotion Recognition - Step 1: Data Exploration & Audio Loading
This script helps you understand your audio data and load it properly.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf

# ============================================================================
# PART 1: UNDERSTAND RAVDESS FILENAME CONVENTION
# ============================================================================
def parse_ravdess_filename(filename):
    """
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 
               05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24, odd = male, even = female)
    """
    parts = filename.replace('.wav', '').split('-')
    
    emotion_dict = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    return {
        'filename': filename,
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': emotion_dict.get(parts[2], 'unknown'),
        'emotion_code': parts[2],
        'intensity': 'normal' if parts[3] == '01' else 'strong',
        'statement': parts[4],
        'repetition': parts[5],
        'actor': int(parts[6]),
        'gender': 'male' if int(parts[6]) % 2 == 1 else 'female'
    }

# ============================================================================
# PART 2: PARSE CREMA-D FILENAME CONVENTION
# ============================================================================
def parse_crema_filename(filename):
    """
    CREMA-D filename format: 1001_DFA_ANG_XX.wav
    - Actor ID (1001-1091)
    - Sentence (IEO, TIE, IOM, IWW, TAI, MTI, IWL, ITH, DFA, ITS, WSI)
    - Emotion (ANG = anger, DIS = disgust, FEA = fear, HAP = happy, 
               NEU = neutral, SAD = sad)
    - Intensity (LO = low, MD = medium, HI = high, XX = unspecified)
    """
    parts = filename.replace('.wav', '').split('_')
    
    emotion_dict = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    return {
        'filename': filename,
        'actor': int(parts[0]),
        'sentence': parts[1],
        'emotion': emotion_dict.get(parts[2], 'unknown'),
        'intensity': parts[3]
    }

# ============================================================================
# PART 3: LOAD AND EXPLORE AUDIO FILES
# ============================================================================
def load_audio_file(file_path, sr=22050):
    """
    Load an audio file and return waveform and sample rate
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (22050 Hz is standard)
    
    Returns:
        audio: Audio time series
        sample_rate: Sample rate of audio
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def visualize_audio_sample(file_path):
    """
    Visualize a single audio sample with waveform and spectrogram
    """
    audio, sr = load_audio_file(file_path)
    
    if audio is None:
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform', fontsize=14)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram', fontsize=14)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # 3. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title('Mel Spectrogram', fontsize=14)
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('audio_visualization_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print audio properties
    print(f"\n{'='*60}")
    print(f"Audio Properties:")
    print(f"{'='*60}")
    print(f"Duration: {len(audio)/sr:.2f} seconds")
    print(f"Sample Rate: {sr} Hz")
    print(f"Total Samples: {len(audio)}")
    print(f"Audio Shape: {audio.shape}")
    print(f"Min Amplitude: {audio.min():.4f}")
    print(f"Max Amplitude: {audio.max():.4f}")
    print(f"Mean Amplitude: {audio.mean():.4f}")

# ============================================================================
# PART 4: CREATE DATASET CATALOG
# ============================================================================
def create_dataset_catalog(data_dir, dataset_type='ravdess'):
    """
    Scan directory and create a catalog of all audio files with metadata
    
    Args:
        data_dir: Path to dataset directory
        dataset_type: 'ravdess' or 'crema'
    
    Returns:
        DataFrame with file information
    """
    file_list = []
    
    # Walk through directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Parse filename based on dataset type
                if dataset_type == 'ravdess':
                    metadata = parse_ravdess_filename(file)
                else:  # crema
                    metadata = parse_crema_filename(file)
                
                metadata['file_path'] = file_path
                file_list.append(metadata)
    
    df = pd.DataFrame(file_list)
    print(f"\nFound {len(df)} audio files in {data_dir}")
    return df

def analyze_dataset_distribution(df):
    """
    Analyze and visualize emotion distribution in dataset
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Emotion distribution
    emotion_counts = df['emotion'].value_counts()
    axes[0].bar(emotion_counts.index, emotion_counts.values, color='skyblue', edgecolor='black')
    axes[0].set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gender distribution (if available)
    if 'gender' in df.columns:
        gender_emotion = pd.crosstab(df['emotion'], df['gender'])
        gender_emotion.plot(kind='bar', ax=axes[1], color=['lightcoral', 'lightblue'])
        axes[1].set_title('Emotion Distribution by Gender', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Emotion')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Gender')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"\nTotal samples: {len(df)}")
    print(f"\nEmotion distribution:")
    print(emotion_counts)
    
    if 'gender' in df.columns:
        print(f"\nGender distribution:")
        print(df['gender'].value_counts())

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    
    print("="*60)
    print("SPEECH EMOTION RECOGNITION - DATA EXPLORATION")
    print("="*60)
    
    # Step 1: Set your data directory paths
    RAVDESS_DIR = "data/ravdess"  # Update this path
    CREMA_DIR = "data/crema_d"     # Update this path
    
    # Step 2: Create catalog for RAVDESS dataset
    print("\n[1] Loading RAVDESS Dataset...")
    if os.path.exists(RAVDESS_DIR):
        ravdess_df = create_dataset_catalog(RAVDESS_DIR, dataset_type='ravdess')
        print("\nFirst few samples:")
        print(ravdess_df.head())
        
        # Analyze distribution
        analyze_dataset_distribution(ravdess_df)
        
        # Visualize a sample audio
        if len(ravdess_df) > 0:
            sample_file = ravdess_df.iloc[0]['file_path']
            print(f"\n[2] Visualizing sample audio: {sample_file}")
            visualize_audio_sample(sample_file)
        
        # Save catalog
        ravdess_df.to_csv('ravdess_catalog.csv', index=False)
        print("\n✓ RAVDESS catalog saved to 'ravdess_catalog.csv'")
    else:
        print(f"Directory not found: {RAVDESS_DIR}")
    
    # Step 3: Create catalog for CREMA-D dataset (optional)
    print("\n[3] Loading CREMA-D Dataset...")
    if os.path.exists(CREMA_DIR):
        crema_df = create_dataset_catalog(CREMA_DIR, dataset_type='crema')
        print("\nFirst few samples:")
        print(crema_df.head())
        
        analyze_dataset_distribution(crema_df)
        
        crema_df.to_csv('crema_catalog.csv', index=False)
        print("\n✓ CREMA-D catalog saved to 'crema_catalog.csv'")
    else:
        print(f"Directory not found: {CREMA_DIR}")
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Check the CSV catalogs for your datasets")
    print("3. Proceed to feature extraction (Step 2)")