import cv2
import numpy as np
from ultralytics import YOLO

def test_onnx_model():
    # Test with Python first to verify the model works
    print("Testing YOLOv8n ONNX model with Python...")
    
    # Load the ONNX model
    model = YOLO('yolov8n.onnx')
    
    # Load a frame from the video
    cap = cv2.VideoCapture('data/cars.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Run inference
    results = model(frame, conf=0.25)
    
    # Print results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Detected {len(boxes)} objects")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  {class_name}: {conf:.3f}")
        else:
            print("No detections")
    
    print("Python test complete")

if __name__ == "__main__":
    test_onnx_model() 