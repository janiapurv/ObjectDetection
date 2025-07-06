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

    float confThreshold = 0.3f;
    float iouThreshold = 0.45f;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Print output shape for debugging (only once)
    static bool firstFrame = true;
    if (firstFrame && !outputs.empty()) {
        std::cout << "Output shape: [";
        for (int i = 0; i < outputs[0].dims; ++i) {
            std::cout << outputs[0].size[i];
            if (i < outputs[0].dims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        firstFrame = false;
    }

    if (!outputs.empty()) {
        cv::Mat output = outputs[0];
        if (output.dims == 3 && output.size[1] == 84) {
            cv::Mat reshaped = output.reshape(1, output.size[2]); // [8400, 84]
            for (int i = 0; i < reshaped.rows; ++i) {
                float* row = reshaped.ptr<float>(i);
                float x_center = row[0];
                float y_center = row[1];
                float width = row[2];
                float height = row[3];

                // Find max class probability (apply sigmoid to all class logits)
                float maxProb = 0.0f;
                int classId = -1;
                for (int c = 4; c < 84; ++c) {
                    float prob = 1.0f / (1.0f + exp(-row[c])); // sigmoid
                    if (prob > maxProb) {
                        maxProb = prob;
                        classId = c - 4;
                    }
                }
                if (maxProb > confThreshold) {
                    // Convert to image coordinates (input is 640x640, scale to frame size)
                    float scaleX = float(frame.cols) / inputWidth;
                    float scaleY = float(frame.rows) / inputHeight;
                    int left = int((x_center - width/2) * scaleX);
                    int top = int((y_center - height/2) * scaleY);
                    int w = int(width * scaleX);
                    int h = int(height * scaleY);
                    left = std::max(0, std::min(left, frame.cols - 1));
                    top = std::max(0, std::min(top, frame.rows - 1));
                    w = std::max(1, std::min(w, frame.cols - left));
                    h = std::max(1, std::min(h, frame.rows - top));
                    if (w > 10 && h > 10) {
                        classIds.push_back(classId);
                        confidences.push_back(maxProb);
                        boxes.emplace_back(left, top, w, h);
                        if (classIds.size() <= 5) {
                            std::cout << "Detection " << classIds.size() << ": "
                                      << classNames[classId] << " conf=" << maxProb
                                      << " box=[" << left << "," << top << "," << w << "," << h << "]" << std::endl;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Total detections before NMS: " << boxes.size() << std::endl;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
    std::cout << "Detections after NMS: " << indices.size() << std::endl;
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float conf = confidences[idx];
        cv::Scalar color;
        switch (classId) {
            case 0: color = cv::Scalar(255, 0, 0); break;   // person - blue
            case 2: color = cv::Scalar(0, 255, 0); break;   // car - green
            case 7: color = cv::Scalar(0, 0, 255); break;   // truck - red
            default: color = cv::Scalar(255, 255, 0); break; // others - cyan
        }
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