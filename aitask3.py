import cv2
import torch
import os

def detect_objects(video_path, custom_classes):
    # Load YOLOv5 model (pretrained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)  # Larger model variant for better accuracy

    # Check if the video file exists
    if not os.path.exists(video_path):
        print("Error: Video file does not exist! Double-check the path.")
        return

    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("Video opened successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of the video

        # Resize the frame for consistent processing
        frame_resized = cv2.resize(frame, (1280, 720))
        h, w, _ = frame_resized.shape

        # Perform YOLOv5 detection
        with torch.no_grad():  # Disable gradients for better performance
            results = model(frame_resized)
        
        detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame

        # Draw detected objects and lane overlays
        def draw_lanes_and_priority(img, detections):
            lane_color = (0, 255, 0)  # Green lanes for active lanes
            inactive_lane_color = (0, 0, 255)  # Red lanes for inactive lanes
            thickness = 2
            
            # Draw horizontal and vertical center lanes
            cv2.line(img, (0, h // 2), (w, h // 2), lane_color, thickness)
            cv2.line(img, (w // 2, 0), (w // 2, h), lane_color, thickness)

            # Vehicle counts in each lane
            lane_counts = {'Up-Left': 0, 'Up-Right': 0, 'Down-Left': 0, 'Down-Right': 0}
            
            for _, row in detections.iterrows():
                class_id = int(row['class'])
                if class_id in custom_classes:
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Assign detected vehicles to a lane
                    if center_x < w // 2 and center_y < h // 2:
                        lane_counts['Up-Left'] += 1
                    elif center_x > w // 2 and center_y < h // 2:
                        lane_counts['Up-Right'] += 1
                    elif center_x < w // 2 and center_y > h // 2:
                        lane_counts['Down-Left'] += 1
                    else:
                        lane_counts['Down-Right'] += 1
                    
                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Determine the priority lane
            max_lane = max(lane_counts, key=lane_counts.get)
            cv2.putText(img, f"Priority Lane: {max_lane} (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw lanes and priority
        draw_lanes_and_priority(frame_resized, detections)

        # Display the result
        cv2.imshow("Real-Time Lane Prioritization", frame_resized)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the video file
    video_path = 'C:\\AITEST1\\video\\carvideo.mp4'  # Replace with the path to your video
    CUSTOM_CLASSES = [2, 3, 5, 7]  # Vehicle classes: car, motorcycle, bus, truck

    detect_objects(video_path, CUSTOM_CLASSES)
