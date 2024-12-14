**Speech Command Recognition Using Machine Learning**

**Overview**

This project implements a machine learning-based speech recognition system capable of identifying absolute language keywords: "all," "must," "never," "none," and "only." It is optimized for real-time inference on embedded devices like the Arduino Nano 33 BLE Sense.

The system has potential applications in mental health monitoring and linguistic analysis, where detecting absolute statements can offer insights into emotional states or stress levels.

**Features**

**Custom Dataset:** 5000 audio clips (~1 second each) with diverse speaker accents and background noise conditions.

**Preprocessing Pipeline:** Converts raw audio to spectrograms using Short-Time Fourier Transform (STFT), with resampling, padding, and normalization.

**Efficient Model Design:** Lightweight CNN architecture suitable for embedded deployment.

**Deployment Optimizations:** Includes post-training quantization to reduce model size and improve efficiency on resource-constrained devices.

**Dataset**
**Keywords:** "all," "must," "never," "none," "only."

**Total Samples:** ~5000 (50 new samples per keyword + ~950 pre-existing samples per keyword).

**Format:** WAV files at 16 kHz, 1-second duration.

**Model Architecture**

The model uses a Convolutional Neural Network (CNN) with the following key layers:

Resizing and normalization for input processing.

Depthwise separable convolution for efficient feature extraction.

MaxPooling and Dropout for regularization and dimensionality reduction.

Fully connected layers for classification.
