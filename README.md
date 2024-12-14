# Speech Command Recognition for Absolute Keywords

## Overview
This project implements a machine learning-based speech recognition system capable of identifying absolute language keywords: **"all," "must," "never," "none,"** and **"only."** It is optimized for real-time inference on embedded devices like the Arduino Nano 33 BLE Sense.

The system has potential applications in mental health monitoring and linguistic analysis, where detecting absolute statements can offer insights into emotional states or stress levels.

## Features
- **Custom Dataset**: 5000 audio clips (~1 second each) with diverse speaker accents and background noise conditions.  
- **Preprocessing Pipeline**: Converts raw audio to spectrograms using Short-Time Fourier Transform (STFT), with resampling, padding, and normalization.  
- **Efficient Model Design**: Lightweight CNN architecture suitable for embedded deployment.  
- **Deployment Optimizations**: Includes post-training quantization to reduce model size and improve efficiency on resource-constrained devices.  

## Dataset
- **Keywords**: "all," "must," "never," "none," "only."  
- **Total Samples**: ~5000 (50 new samples per keyword + ~950 pre-existing samples per keyword).  
- **Format**: WAV files at 16 kHz, 1-second duration.  

## Model Architecture
The model uses a **Convolutional Neural Network (CNN)** with the following key layers:
- Resizing and normalization for input processing.
- Depthwise separable convolution for efficient feature extraction.
- MaxPooling and Dropout for regularization and dimensionality reduction.
- Fully connected layers for classification.

### Key Metrics
- **Training Accuracy**: ~89%  
- **Validation Accuracy**: ~94%  

## Training Details
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Batch Size**: 64  
- **Epochs**: 10 (with early stopping based on validation loss)  
- **Learning Rate**: 0.001  

## Deployment
The model has been optimized for deployment on embedded systems:
1. **Quantization**: Reduced weights and activations to 8-bit integers for faster and energy-efficient inference.
2. **Target Device**: Arduino Nano 33 BLE Sense.

## Preprocessing Pipeline
1. Load audio files with TensorFlow utilities.  
2. Resample to 16 kHz.  
3. Pad or truncate to 1 second.  
4. Generate spectrograms using STFT.  
5. Normalize the spectrograms for consistency.  

## How to Use
### Requirements
- Python 3.7+  
- TensorFlow 2.x  
- NumPy  
- Matplotlib  

### Steps
1. **Prepare Dataset**: Place audio files in the required format and structure.  
2. **Preprocess Data**: Use the provided scripts to generate spectrograms and prepare data for training.  
3. **Train Model**: Adjust hyperparameters and train the model using `train.py`.  
4. **Quantize Model**: Use TensorFlow Lite tools to apply post-training quantization.  
5. **Deploy**: Flash the quantized model to your target device.  

## Results
- Achieved **94% validation accuracy** with high robustness to noise.
- Deployment-friendly design ensures smooth inference on embedded systems.

## Applications
- **Mental Health Monitoring**: Identifying stress or emotional cues in speech.  
- **Linguistic Analysis**: Detecting the use of absolute language in communication.  
- **Smart Devices**: Voice-activated systems with a specific command set.  

## License
[MIT License](LICENSE)
