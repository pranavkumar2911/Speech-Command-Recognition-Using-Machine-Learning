Speech Command Recognition for Absolute Keywords
This project implements a speech command recognition system designed to classify spoken instances of specific absolute keywords: "all," "must," "never," "none," and "only." The system leverages machine learning, particularly a Convolutional Neural Network (CNN) model trained on spectrogram data, to achieve robust keyword detection.

Key Features:
Custom Dataset: Includes 5000 audio clips (~1 second each) with diverse accents and background noise.

Preprocessing Pipeline: Converts raw audio to spectrograms using Short-Time Fourier Transform (STFT), with resampling, padding, and normalization.

Model Design: A lightweight CNN optimized for embedded systems.

Deployment Ready: Post-training quantization enables real-time inference on devices like the Arduino Nano 33 BLE Sense.

Performance: Achieved ~94% validation accuracy with a focus on noise robustness and real-world generalization.

This system is tailored for applications in mental health monitoring and other domains requiring accurate detection of absolute language in speech.
