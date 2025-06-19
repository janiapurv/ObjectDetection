#include "object_detector.h"
#include <iostream>
#include <fstream>
#include <algorithm>

ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& videoPath)
    : playbackSpeed(0.5f)
{
    net = cv::dnn::readNet(modelPath);
    cap.open(videoPath);
    if (!cap.isOpened()) throw std::runtime_error("Could not open video file: " + videoPath);
    loadClassNames();
}

void ObjectDetector::loadClassNames() {
    // COCO 80 classes
    classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}

void ObjectDetector::processFrame(cv::Mat& frame) {
    int inputWidth = 640, inputHeight = 640;
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float confThreshold = 0.5f;
    float iouThreshold = 0.45f;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Print output shape for debugging
    if (!outputs.empty()) {
        std::cout << "Output shape: [";
        for (int i = 0; i < outputs[0].dims; ++i) {
            std::cout << outputs[0].size[i];
            if (i < outputs[0].dims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Typical YOLOv8 ONNX output: [1, N, 84]
    if (!outputs.empty() && outputs[0].dims == 3) {
        const int numDetections = outputs[0].size[1];
        const int numElements = outputs[0].size[2];
        float* data = (float*)outputs[0].data;
        for (int i = 0; i < numDetections; ++i) {
            float objConf = data[4];
            if (objConf < confThreshold) {
                data += numElements;
                continue;
            }
            float* classScores = data + 5;
            cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
            cv::Point classIdPoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
            float confidence = objConf * (float)maxClassScore;
            if (confidence > confThreshold) {
                float cx = data[0] * frame.cols;
                float cy = data[1] * frame.rows;
                float w = data[2] * frame.cols;
                float h = data[3] * frame.rows;
                int left = int(cx - w / 2);
                int top = int(cy - h / 2);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(confidence);
                boxes.emplace_back(left, top, int(w), int(h));
            }
            data += numElements;
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float conf = confidences[idx];
        cv::Scalar color = cv::Scalar(0, 255, 0); // You can randomize per class if you want
        cv::rectangle(frame, box, color, 2);
        std::string label = classNames[classId] + " " + cv::format("%.2f", conf);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
                      cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
        cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }
}

void ObjectDetector::run() {
    cv::Mat frame;
    while (cap.read(frame)) {
        processFrame(frame);
        cv::putText(frame, "Press 'q' to quit, '+' to speed up, '-' to slow down", {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        cv::putText(frame, "Current speed: " + std::to_string(playbackSpeed) + "x", {10, 70},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        cv::imshow("YOLO Object Detection", frame);
        int delay = static_cast<int>(1000 / (cap.get(cv::CAP_PROP_FPS) * playbackSpeed));
        char key = (char)cv::waitKey(delay);
        if (key == 'q') break;
        else if (key == '+' || key == '=') playbackSpeed = std::min(2.0f, playbackSpeed + 0.1f);
        else if (key == '-' || key == '_') playbackSpeed = std::max(0.1f, playbackSpeed - 0.1f);
    }
    cap.release();
    cv::destroyAllWindows();
} 