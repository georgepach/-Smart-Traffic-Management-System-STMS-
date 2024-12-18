import cv2
import numpy as np
import torch

# Load YOLOv5 model (ensure YOLOv5 weights are in the directory)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define Regions of Interest (ROI) based on the image (1920x1080 resolution)
def define_rois(frame):
    # Define ROIs as per the given coordinates
    Forward_Left = np.array([[0, 470], [200, 470], [300, 300], [0, 300]])
    Forward_Up = np.array([[300, 300], [600, 300], [600, 150], [300, 150]])
    Forward_Right = np.array([[700, 470], [900, 470], [900, 400], [700, 400]])

    # Store the regions in a dictionary
    rois = {
        "Forward_Left": Forward_Left,
        "Forward_Up": Forward_Up,
        "Forward_Right": Forward_Right
    }
    return rois

# Draw ROIs on the frame
def draw_rois(frame, rois):
    for key, points in rois.items():
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, key, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# Detect vehicles and count them in each ROI
def detect_and_count(frame, rois):
    results = model(frame)
    vehicles = ['car', 'truck', 'bus', 'motorbike']
    counts = {key: 0 for key in rois}

    # Extract detection results
    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in vehicles:
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check which ROI the vehicle falls into
            for key, points in rois.items():
                if cv2.pointPolygonTest(points, (center_x, center_y), False) >= 0:
                    counts[key] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, model.names[int(cls)], (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame, counts

# Determine priority lane based on vehicle counts
def determine_priority(counts):
    priority = max(counts, key=counts.get)
    return priority

# Main function to process the video
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rois = define_rois(frame)
        frame = draw_rois(frame, rois)
        frame, counts = detect_and_count(frame, rois)

        # Determine and display the priority lane
        priority_lane = determine_priority(counts)
        cv2.putText(frame, f"Priority Lane: {priority_lane}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display the processed frame
        cv2.imshow('Traffic Priority Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function with your video file
if __name__ == "__main__":
    video_path = "C:\\AITEST1\\video\\carvideo.mp4"
    main(video_path)
