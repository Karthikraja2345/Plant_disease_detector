# 🌿 Smart Leaf Disease Detector (OpenCV Project)

A real-time crop disease detection system built with **OpenCV** and a pre-trained Hugging Face MobileNetV2 model. Identify 38+ plant disease classes or healthy leaves from live webcam feed or static images, with voice feedback and saved results.

## 🚀 Features

- 🖼️ **Core OpenCV Processing**: Captures and preprocesses frames/images using OpenCV for leaf gating and annotation
- 🤖 **Hugging Face Model**: Uses `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` for 38+ classes at ~95.4% accuracy
- 📹 **Live Webcam & Image Modes**: Press SPACE to analyze live feed or provide an image file
- 🍃 **Leaf Gating**: Ensures only frames with ≥20% green leaf coverage are analyzed
- 🔝 **Top 3 Predictions**: Displays and announces the top 3 disease or healthy labels with confidence scores
- 🔊 **Voice Feedback**: Announces detection results via text-to-speech
- 💾 **Result Saving**: Annotated detection images saved to `detections/` with timestamp
- 🔧 **Pure OpenCV Integration**: All image capture, masking, and drawing use OpenCV functions

## 📦 Installation

1. Clone the repository or copy the code files.
2. Install dependencies:
   ```bash
   pip install opencv-python numpy pillow transformers torch torchvision pyttsx3
   ```

## 🎯 Usage

1. Run the script:
   ```bash
   python smart_leaf_detector.py
   ```
2. Choose mode when prompted:
   - `1` – Live Webcam
   - `2` – Image File
3. **Webcam Mode**:
   - Press **SPACE** to capture and analyze the frame.
   - Voice and console output announce the top disease prediction.
   - Annotated result displays and saves to `detections/`.
   - Press **q** to quit.
4. **Image File Mode**:
   - Enter the path to a leaf image file.
   - Analysis runs once, voice and console output announce the result.
   - Annotated image saves to `detections/`.
   - Press any key to close the window.

## 📂 Directory Structure

```
.
├── leaf_disease_detector.py    # Main OpenCV + Hugging Face script
├── detections/               # Saved annotated images
└── README.md                 # Project documentation
```

## ⚙️ Customization

- **Change Model**
  The script uses the MobileNetV2 plant disease model by default. To use a different Hugging Face model, update in the constructor:
  ```python
  self.classifier = pipeline(
      "image-classification",
      model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
  )
  ```
- **Leaf Coverage Threshold**
  Adjust `self.min_leaf_coverage` to change the required green coverage (default 0.20).
- **Voice Rate**
  Modify `self.tts.setProperty('rate', 150)` for different speech speed.
