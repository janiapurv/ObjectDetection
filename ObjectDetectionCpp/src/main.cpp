#include "object_detector.h"
#include <iostream>

int main() {
    try {
        std::string modelPath = "data/yolov8n.onnx";
        std::string videoPath = "data/cars.mp4";
        ObjectDetector detector(modelPath, videoPath);
        detector.run();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 