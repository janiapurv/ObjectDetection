import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

class ObjectDetector:
    def __init__(self, video_path):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Ensure video path is absolute
        if not os.path.isabs(video_path):
            # If relative path, assume it's relative to the data directory
            video_path = os.path.join('data', video_path)
        
        # Initialize video source
        if not os.path.exists(video_path):
            raise ValueError(f"Video file {video_path} not found! Please ensure the video file exists in the data directory.")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Set playback speed (lower = slower)
        self.playback_speed = 0.5  # 0.5x speed (half speed)
        
        print(f"Video properties: {self.width}x{self.height} @ {self.fps}fps")
        print(f"Playback speed: {self.playback_speed}x")
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        
    def process_frame(self, frame):
        # Run YOLO inference
        results = self.model(frame, conf=0.5)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name
                class_name = self.model.names[cls]
                
                # Draw bounding box
                color = tuple(map(int, self.colors[cls]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run(self):
        try:
            while True:
                # Read frame from video
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video file")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate delay based on playback speed
                delay = int(1000 / (self.fps * self.playback_speed))  # Convert to milliseconds
                
                # Display instructions
                cv2.putText(processed_frame, f'Press "q" to quit, "+" to speed up, "-" to slow down',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Current speed: {self.playback_speed:.1f}x',
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('YOLO Object Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):  # Speed up
                    self.playback_speed = min(2.0, self.playback_speed + 0.1)
                elif key == ord('-') or key == ord('_'):  # Slow down
                    self.playback_speed = max(0.1, self.playback_speed - 0.1)
                
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # Video file name (will be looked for in the data directory)
        video_file = 'cars.mp4'
        detector = ObjectDetector(video_file)
        detector.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}") 