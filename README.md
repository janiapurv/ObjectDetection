# Object Detection Project

This repository contains two implementations of object detection on video using YOLOv8:

- **Python**: See `object_detection.py` (uses Ultralytics YOLOv8 Python API)
- **C++**: See the `ObjectDetectionCpp/` directory (uses OpenCV DNN with YOLOv8 ONNX export)

## C++ Implementation (ObjectDetectionCpp)

### Requirements
- OpenCV (with DNN module)
- CMake
- C++17 compiler
- `yolov8n.onnx` model file (exported from Python, place in `ObjectDetectionCpp/data/`)
- `cars.mp4` video file (place in `ObjectDetectionCpp/data/`)

### Build
```sh
cd ObjectDetectionCpp
mkdir build
cd build
cmake ..
cmake --build .
```

### Run
```sh
./ObjectDetectionCpp
```

## Features

- Object detection using YOLOv8
- Video file processing
- Adjustable playback speed
- Real-time visualization with bounding boxes and labels
- Support for multiple video formats

## Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- Ultralytics YOLO

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd ObjectDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your video files in the `data` directory
2. Run the object detection script:
```bash
python object_detection.py
```

### Controls
- Press '+' to increase playback speed
- Press '-' to decrease playback speed
- Press 'q' to quit

## Project Structure

```
ObjectDetection/
├── data/               # Directory for video files
├── object_detection.py # Main script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 