import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video
# file_name = 'ball.mp4'
file_name = 'objectTracking_examples_multiObject.avi'
cap = cv2.VideoCapture(file_name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))



# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = [0, 0, 0, 0]  # State vector: [x, y, dx, dy]
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])  # State transition matrix
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])  # Observation matrix
kf.R *= 10  # Measurement noise covariance
kf.Q *= 0.1  # Process noise covariance
kf.P *= 100  # State uncertainty covariance


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for *xyxy, conf, cls in results.xyxy[0].tolist():
        # detect ball only
        if cls == 32:
            detected_ball = True

            label = f'{results.names[int(cls)]}: {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            centroid_x = int((xyxy[0] + xyxy[2]) / 2)
            centroid_y = int((xyxy[1] + xyxy[3]) / 2)
            frame = cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

        if detected_ball:
            # Predict the next centroid position using the Kalman filter
            kf.predict()
            predicted_centroid_x, predicted_centroid_y = int(kf.x[0]), int(kf.x[1])

            # Draw the predicted centroid
            frame = cv2.circle(frame, (predicted_centroid_x, predicted_centroid_y), 5, (0, 255, 255), -1)
    # Display the frame
    cv2.imshow('Video', frame)

    out.write(frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
