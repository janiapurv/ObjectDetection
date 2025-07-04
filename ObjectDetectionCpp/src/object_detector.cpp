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

    float confThreshold = 0.25f;  // Lower threshold for better detection
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

    // YOLOv8 ONNX output parsing
    if (!outputs.empty()) {
        cv::Mat output = outputs[0];
        
        // Handle different output formats
        if (output.dims == 3) {
            // Format: [1, N, 84] - typical YOLOv8 ONNX output
            const int numDetections = output.size[1];
            const int numElements = output.size[2];
            float* data = (float*)output.data;
            
            std::cout << "Processing " << numDetections << " detections with " << numElements << " elements each" << std::endl;
            
            for (int i = 0; i < numDetections; ++i) {
                // Get class scores (skip first 4 elements which are bbox coords)
                float* classScores = data + 4;
                cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
                cv::Point classIdPoint;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
                
                float confidence = (float)maxClassScore;
                if (confidence > confThreshold) {
                    // Get bounding box coordinates (normalized to 0-1)
                    float x_center = data[0];
                    float y_center = data[1];
                    float width = data[2];
                    float height = data[3];
                    
                    // Convert to image coordinates
                    int left = int((x_center - width/2) * frame.cols);
                    int top = int((y_center - height/2) * frame.rows);
                    int w = int(width * frame.cols);
                    int h = int(height * frame.rows);
                    
                    // Ensure coordinates are within image bounds
                    left = std::max(0, left);
                    top = std::max(0, top);
                    w = std::min(w, frame.cols - left);
                    h = std::min(h, frame.rows - top);
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(confidence);
                    boxes.emplace_back(left, top, w, h);
                    
                    // Debug output for first few detections
                    if (classIds.size() <= 3) {
                        std::cout << "Detection " << classIds.size() << ": " 
                                  << classNames[classIdPoint.x] << " conf=" << confidence 
                                  << " box=[" << left << "," << top << "," << w << "," << h << "]" << std::endl;
                    }
                }
                data += numElements;
            }
        } else if (output.dims == 2) {
            // Format: [N, 84] - alternative YOLOv8 ONNX output
            const int numDetections = output.size[0];
            const int numElements = output.size[1];
            float* data = (float*)output.data;
            
            std::cout << "Processing " << numDetections << " detections with " << numElements << " elements each" << std::endl;
            
            for (int i = 0; i < numDetections; ++i) {
                float* classScores = data + 4;
                cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
                cv::Point classIdPoint;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
                
                float confidence = (float)maxClassScore;
                if (confidence > confThreshold) {
                    float x_center = data[0];
                    float y_center = data[1];
                    float width = data[2];
                    float height = data[3];
                    
                    int left = int((x_center - width/2) * frame.cols);
                    int top = int((y_center - height/2) * frame.rows);
                    int w = int(width * frame.cols);
                    int h = int(height * frame.rows);
                    
                    left = std::max(0, left);
                    top = std::max(0, top);
                    w = std::min(w, frame.cols - left);
                    h = std::min(h, frame.rows - top);
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(confidence);
                    boxes.emplace_back(left, top, w, h);
                    
                    // Debug output for first few detections
                    if (classIds.size() <= 3) {
                        std::cout << "Detection " << classIds.size() << ": " 
                                  << classNames[classIdPoint.x] << " conf=" << confidence 
                                  << " box=[" << left << "," << top << "," << w << "," << h << "]" << std::endl;
                    }
                }
                data += numElements;
            }
        } else {
            std::cout << "Unexpected output dimensions: " << output.dims << std::endl;
        }
    }

    // Print detection count for debugging
    if (!boxes.empty()) {
        std::cout << "Total detections before NMS: " << boxes.size() << std::endl;
    } else {
        std::cout << "No detections found" << std::endl;
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
    
    std::cout << "Detections after NMS: " << indices.size() << std::endl;
    
    // Draw detections
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float conf = confidences[idx];
        
        // Use different colors for different classes
        cv::Scalar color;
        if (classId == 2) { // car class
            color = cv::Scalar(0, 255, 0); // green for cars
        } else {
            color = cv::Scalar(255, 0, 0); // blue for others
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