# YOLO Object Detection

This project implements real-time object detection using YOLO (You Only Look Once) and OpenCV. It can process video files and detect various objects in them.

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