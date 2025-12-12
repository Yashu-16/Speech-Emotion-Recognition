Speech Emotion Recognition
A comprehensive machine learning project that recognizes human emotions from speech using both classical ML and deep learning approaches.
👥 Team Members
Yash Randhe

📋 Project Overview
This project implements a Speech Emotion Recognition system capable of detecting emotions such as happiness, anger, sadness, and neutrality from audio recordings. It uses multiple approaches including:

Classical Machine Learning (SVM, Random Forest)
Deep Learning (CNN, Feedforward Neural Networks)
Ensemble Methods (Voting, Weighted Fusion)

🎯 Features

✅ Extract 74 acoustic features (MFCCs, pitch, energy, spectral features)
✅ Train multiple ML models and compare performance
✅ Implement ensemble techniques for improved accuracy
✅ Generate comprehensive visualizations (spectrograms, t-SNE, confusion matrices)
✅ Achieve 75-90% accuracy with ensemble methods

📊 Datasets

RAVDESS: 1,440 audio files from 24 actors expressing 8 emotions
CREMA-D: 7,442 clips from 91 actors expressing 6 emotions

Download Datasets:

RAVDESS: https://zenodo.org/record/1188976
CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D

🚀 Installation
Prerequisites

Python 3.8 or higher
pip package manager

Setup

Clone the repository:

bashgit clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition

Create virtual environment:

bash# Windows
python -m venv env
.\env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate

Install dependencies:

bashpip install -r requirements.txt

Download datasets:

Download RAVDESS and extract to data/ravdess/
Download CREMA-D (AudioWAV) and extract to data/crema_d/AudioWAV/



📁 Project Structure
speech-emotion-recognition/
├── data/
│   ├── ravdess/              # RAVDESS dataset
│   └── crema_d/              # CREMA-D dataset
├── step1_data_exploration.py
├── step2_feature_extraction.py
├── step3_classical_ml.py
├── step4_deep_learning.py
├── step5_fusion_visualization.py
├── requirements.txt
├── .gitignore
└── README.md
🎮 Usage
Run the scripts in order:
Step 1: Data Exploration
bashpython step1_data_exploration.py

Loads and visualizes audio files
Creates dataset catalogs
Generates spectrograms

Step 2: Feature Extraction
bashpython step2_feature_extraction.py

Extracts 74 acoustic features per audio file
Saves features to CSV files
Visualizes feature distributions

Step 3: Classical ML Models
bashpython step3_classical_ml.py

Trains SVM and Random Forest models
Generates confusion matrices
Compares model performance

Step 4: Deep Learning Models
bashpython step4_deep_learning.py

Trains CNN and Feedforward Neural Networks
Implements early stopping and checkpointing
Plots training history

Step 5: Model Fusion & Visualization
bashpython step5_fusion_visualization.py

Implements ensemble methods
Creates t-SNE and PCA visualizations
Generates final comparison report

📈 Results
ModelAccuracySVM60-75%Random Forest65-80%Feedforward NN65-80%CNN70-85%Ensemble (Hard Voting)75-88%Ensemble (Soft Voting)75-90%
🔬 Methodology
Feature Extraction

MFCCs (52 features): Mel-frequency cepstral coefficients
Pitch (4 features): Fundamental frequency statistics
Energy (4 features): RMS energy measures
Spectral (6 features): Centroid, bandwidth, rolloff
ZCR (4 features): Zero-crossing rate
Chroma (4 features): Pitch class profiles

Model Architectures
Feedforward Neural Network:
Input (74) → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Output
1D CNN:
Input → Conv1D(64) → Conv1D(128) → Conv1D(256) → Dense(128) → Output
📊 Visualizations
The project generates:

Waveforms and spectrograms
Confusion matrices for all models
t-SNE embeddings of feature space
PCA projections
Training history curves
Model comparison charts

🛠️ Technologies Used

Python 3.x
TensorFlow/Keras: Deep learning models
Scikit-learn: Classical ML models
Librosa: Audio processing
NumPy & Pandas: Data manipulation
Matplotlib & Seaborn: Visualization

🎓 Academic Context
This project was developed for CPE646 Pattern Recognition course.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

RAVDESS dataset creators
CREMA-D dataset creators
Librosa development team
TensorFlow/Keras team
