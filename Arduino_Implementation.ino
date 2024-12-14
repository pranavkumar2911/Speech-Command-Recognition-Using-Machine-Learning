#include "model.h" // Include your trained model header file
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "Arduino_BMI270_BMM150.h"   // Corrected IMU sensor type
#include <PDM.h> // Library for audio input

// Buffer size and variables
constexpr int kTensorArenaSize = 35 * 1024; // Adjust as per available memory
uint8_t tensor_arena[kTensorArenaSize];
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Audio input
static const char kChannels = 1;
static const int kFrequency = 16000;
short sampleBuffer[512];
volatile int samplesRead = 0;

// Noise and VAD
const int kInitialSilenceThreshold = 1000; // Starting threshold
int dynamicSilenceThreshold = kInitialSilenceThreshold;
const int kNoiseAdjustmentWindow = 20; // Rolling window for noise adjustment
int noiseWindow[kNoiseAdjustmentWindow];
int noiseIndex = 0;

// Command detection constants
const float kCommandThreshold = 0.6f; // Confidence threshold for detection
const char* kCommandLabels[] = {"all", "must", "never", "none", "only"}; // Update based on your model labels

// Callback function for audio input
void onAudioReceived() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable > 0) {
    PDM.read(sampleBuffer, bytesAvailable);
    samplesRead = bytesAvailable / 2; // 16-bit samples
  }
}

void setup() {
  // Initialize serial for debugging
  Serial.begin(115200);
  while (!Serial);

  // Initialize PDM audio
  PDM.onReceive(onAudioReceived);
  if (!PDM.begin(kChannels, kFrequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // Set up error reporting
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model and verify
  const tflite::Model* model = tflite::GetModel(model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch!");
    while (true);
  }

  // Set up all ops resolver and interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Failed to allocate tensors!");
    while (true);
  }

  // Get model input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize noise suppression
  for (int i = 0; i < kNoiseAdjustmentWindow; ++i) {
    noiseWindow[i] = kInitialSilenceThreshold;
  }

  Serial.println("Setup complete!");
}

// Detect speech and adjust noise threshold
bool detectSpeech(short* buffer, int length) {
  int energy = 0;
  for (int i = 0; i < length; i++) {
    energy += abs(buffer[i]);
  }
  energy /= length; // Compute average energy

  // Update dynamic noise threshold
  noiseWindow[noiseIndex] = energy;
  noiseIndex = (noiseIndex + 1) % kNoiseAdjustmentWindow;

  int avgNoise = 0;
  for (int i = 0; i < kNoiseAdjustmentWindow; ++i) {
    avgNoise += noiseWindow[i];
  }
  avgNoise /= kNoiseAdjustmentWindow;
  dynamicSilenceThreshold = avgNoise + 500; // Add buffer above average noise

  return energy > dynamicSilenceThreshold;
}

// Run inference
void loop() {
  // Wait for audio samples
  if (samplesRead > 0) {
    // Detect if speech is present
    bool isSpeaking = detectSpeech(sampleBuffer, samplesRead);

    if (isSpeaking) {
      Serial.println("Speech detected, running inference...");

      // Prepare input tensor
      for (int i = 0; i < input->bytes; ++i) {
        input->data.int8[i] = static_cast<int8_t>(sampleBuffer[i % samplesRead] >> 8); // Convert to int8
      }

      // Clear the samplesRead
      samplesRead = 0;

      // Run inference
      if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Failed to invoke TFLite interpreter!");
        return;
      }

      // Analyze the output to detect the command
      float maxConfidence = 0.0f;
      int detectedCommand = -1;
      for (int i = 0; i < output->dims->data[1]; ++i) {
        float confidence = static_cast<float>(output->data.int8[i]) / 127.0f;
        if (confidence > maxConfidence) {
          maxConfidence = confidence;
          detectedCommand = i;
        }
      }

      // Check if the detected command is valid
      if (maxConfidence > kCommandThreshold) {
        Serial.print("Detected Command: ");
        Serial.print(kCommandLabels[detectedCommand]);
        Serial.print(" (Confidence: ");
        Serial.print(maxConfidence * 100, 2);
        Serial.println("%)");
      } else {
        Serial.println("No valid command detected.");
      }
    }
  }

  delay(300); // Short delay for audio processing
}