from ultralytics import YOLO

if __name__ == "__main__":
    print("Loading YOLOv8n PyTorch model...")
    model = YOLO('yolov8n.pt')
    print("Exporting to ONNX (static input size 640, for OpenCV DNN)...")
    model.export(format='onnx', imgsz=640, simplify=True)
    print("Export complete. ONNX file saved as yolov8n.onnx.") 