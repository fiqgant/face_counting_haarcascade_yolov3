# Face Counting System

This repository contains a Python implementation of a face counting system using Haarcascade and YOLO-Face models. The system detects faces in a video and displays the number of detected faces on each frame. The detection is optimized using Non-Maximum Suppression (NMS) to reduce redundant bounding boxes.

## Features

- **Haarcascade Face Detection**: Detects faces using the Haarcascade classifier.
- **YOLO-Face Detection**: Uses YOLO-Face model to detect faces.
- **Non-Maximum Suppression**: Reduces redundant bounding boxes for better accuracy.
- **Face Counting**: Displays the number of detected faces on each frame.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fiqgant/face_counting_haarcascade_yolov3.git
   cd face-counting-system
   ```

2. **Install the required packages:**

   ```bash
   pip install opencv-python opencv-python-headless numpy
   ```

3. **Download YOLO-Face model files:**

   - Download the YOLO-Face weights file (`yolov3-wider_16000.weights`) and configuration file (`yolov3-face.cfg`).
   - Place these files in a directory named `models` within the project directory.

## Usage

1. **Run the face counting script:**

   ```bash
   python main.py
   ```

2. **Press `q` to quit the video window.**

## Example

The system reads a video file (`video.mp4`) and detects faces in real-time. It uses Haarcascade for initial detection and YOLO-Face for more accurate detection. The system displays the number of detected faces on each frame.

## Code Explanation

- **Haarcascade Initialization:**

  ```python
  haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  ```

- **YOLO-Face Initialization:**

  ```python
  net = cv2.dnn.readNet("models/yolov3-wider_16000.weights", "models/yolov3-face.cfg")
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  ```

- **Main Loop:**

  The main loop reads frames from the video, resizes them for faster processing, and performs face detection using both Haarcascade and YOLO-Face. Detected faces are displayed with bounding boxes, and the number of detected faces is shown on the frame.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
