# ObjectDetectionCpp

C++ implementation of YOLOv8 object detection on video using OpenCV DNN and ONNX export.

## Requirements
- OpenCV (with DNN module)
- CMake
- C++17 compiler
- `yolov8n.onnx` model file (exported from Python, place in `data/`)
- `cars.mp4` video file (place in `data/`)

## Build
```sh
mkdir build
cd build
cmake ..
cmake --build .
```

## Run
```sh
./ObjectDetectionCpp
```

## Notes
- The ONNX output parsing in `processFrame` is a placeholder. You may need to adapt it for your exported YOLOv8 model.
- For best results, use OpenCV 4.5+ with DNN module enabled. 