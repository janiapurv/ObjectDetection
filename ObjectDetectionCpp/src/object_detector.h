#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, const std::string& videoPath);
    void run();

private:
    cv::dnn::Net net;
    cv::VideoCapture cap;
    float playbackSpeed;
    std::vector<std::string> classNames;
    void processFrame(cv::Mat& frame);
    void loadClassNames();
}; 