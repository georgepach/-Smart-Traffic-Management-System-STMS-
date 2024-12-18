import cv2
import numpy as np
import torch

# Load YOLOv5 model (ensure YOLOv5 weights are in the directory)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s' or your custom model

# Define Regions of Interest (ROI)
def define_rois(frame):
    h, w, _ = frame.shape
    rois = {
        "Left": [(0, h//2), (w//2, h)],  # Left lane region
        "Up": [(w//2, 0), (w, h//2)],    # Upward lane region
        "Right": [(w//2, h//2), (w, h)],  # Right lane region
        "Down": [(0, 0), (w//2, h//2)]   # Downward lane region
    }
    return rois

# Draw ROIs on the frame
def draw_rois(frame, rois):
    for key, points in rois.items():
        cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 2)
        cv2.putText(frame, key, (points[0][0] + 10, points[0][1] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# Detect vehicles and count them in each ROI
def detect_and_count(frame, rois):
    results = model(frame)  # Run YOLOv5 detection
    vehicles = ['car', 'truck', 'bus', 'motorbike']
    counts = {key: 0 for key in rois}  # Initialize vehicle counts per ROI

    # Extract detection results
    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in vehicles:  # Only count vehicles
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check which ROI the vehicle falls into
            for key, points in rois.items():
                if points[0][0] <= center_x <= points[1][0] and points[0][1] <= center_y <= points[1][1]:
                    counts[key] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, model.names[int(cls)], (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame, counts

# Determine priority lane
def determine_priority(counts):
    priority = max(counts, key=counts.get)  # Lane with the highest vehicle count
    return priority

# Main function
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Define ROIs and process the frame
        rois = define_rois(frame)
        frame = draw_rois(frame, rois)
        frame, counts = detect_and_count(frame, rois)
        
        # Determine and display the priority lane
        priority_lane = determine_priority(counts)
        cv2.putText(frame, f"Priority Lane: {priority_lane} (Green)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display the processed frame
        cv2.imshow('Traffic Priority Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function with your video file
if __name__ == "__main__":
    video_path = "C:\\AITEST1\\video\\carvideo.mp4"  # Replace with your video file
    main(video_path)
