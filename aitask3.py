import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def detect_objects(video_path, custom_classes):
    # Load YOLOv5 model (pretrained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)  # Use a larger model variant for better accuracy

    # Check if video file exists
    if not os.path.exists(video_path):
        print("Error: Video file does not exist! Double-check the path.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("Video opened successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        # Resize frame for processing
        frame_resized = cv2.resize(frame, (1280, 720))
        h, w, _ = frame_resized.shape

        # Detect objects using YOLOv5 with AMP (Automatic Mixed Precision)
        with torch.amp.autocast('cuda'):
            results = model(frame_resized)
        
        detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame

        # Function to draw overlay for detected objects
        for _, row in detections.iterrows():
            class_id = int(row['class'])  # Get the class ID
            if class_id in custom_classes:  # Check if the detected object belongs to custom classes
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence = row['confidence']  # Confidence score for the detection
                
                # Draw bounding box and label with confidence
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
                cv2.putText(frame_resized, f"Class {class_id} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Function to draw lane overlay and priority indicator
        def draw_lanes_and_priority(img, detections):
            lane_color = (0, 255, 0)  # Green lanes for active lanes
            inactive_lane_color = (0, 0, 255)  # Red lanes for inactive lanes
            thickness = 2
            
            # Draw horizontal center lane
            cv2.line(img, (0, h // 2), (w, h // 2), lane_color, thickness)
            
            # Draw vertical center lane
            cv2.line(img, (w // 2, 0), (w // 2, h), lane_color, thickness)

            # Diagonal lanes for two-way movement
            cv2.line(img, (0, 0), (w, h), lane_color, thickness)
            cv2.line(img, (0, h), (w, 0), lane_color, thickness)
            
            # Determine which lane should be green based on vehicle count
            lane_counts = {'Up-Left': 0, 'Up-Right': 0, 'Down-Left': 0, 'Down-Right': 0}
            for _, row in detections.iterrows():
                x, y = (int(row['xmin']) + int(row['xmax'])) // 2, (int(row['ymin']) + int(row['ymax'])) // 2
                if x < w // 2 and y < h // 2:
                    lane_counts['Up-Left'] += 1
                elif x > w // 2 and y < h // 2:
                    lane_counts['Up-Right'] += 1
                elif x < w // 2 and y > h // 2:
                    lane_counts['Down-Left'] += 1
                else:
                    lane_counts['Down-Right'] += 1
            
            # Find the lane with the highest count for green light
            max_lane = max(lane_counts, key=lane_counts.get)
            
            # Display lane as green if it should be active, else red
            if lane_counts[max_lane] > 0:
                lane_color = (0, 255, 0) if max_lane in ['Up-Left', 'Up-Right'] else (0, 0, 255)
                cv2.putText(img, f"{max_lane} Green", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lane_color, 2)
            else:
                cv2.putText(img, "No active lane", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw lanes and lane priority indicator
        draw_lanes_and_priority(frame_resized, detections)

        # Display the result using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set the video path
    video_path = 'C:\\AITEST1\\video\\carvideo.mp4'  # Set the specific path for your video file
    CUSTOM_CLASSES = [2, 3, 5, 7]  # Replace with your classes; for example, 1 = apple, 2 = banana, 3 = orange

    detect_objects(video_path, CUSTOM_CLASSES)
