# Amharic Sign Language Recognition

This project is an Amharic sign language recognition system that uses a webcam to detect hand gestures and predict corresponding Amharic letters in real-time. The project leverages **Mediapipe** for hand landmark detection, **OpenCV** for capturing video frames and drawing bounding boxes, and a **Random Forest Classifier** trained on hand landmarks for letter classification.


## Features

- Real-time hand gesture recognition using a webcam.
- Prediction of Amharic letters based on hand landmarks.
- Custom Amharic font rendering using the **PIL** library to display predictions on the screen.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.x
- Mediapipe (`mediapipe` library)
- OpenCV (`opencv-python`)
- PIL (Pillow)
- NumPy
- Scikit-learn
- A trained classifier model stored in a `.p` file (Pickle format)

You can install the required Python packages with:

```bash
pip install mediapipe opencv-python pillow numpy scikit-learn
```

## installation

- git clone https://github.com/Naoldaba/Amharic-SignLanguageRecognition.git
- cd Amharic-SignLanguageRecognition

## Running the Model

- python sign_language_predictor.py
