import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter

# Use pre-trained Yolo V5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

file_name = 'ball.mp4'
# file_name = 'objectTracking_examples_multiObject.avi'
cap = cv2.VideoCapture(file_name)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))


# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
# [x, y, dx, dy]
kf.x = np.array([0, 0, 0, 0])
# Transition matrix
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
# Observation matrix
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

kf.R *= 10
kf.Q *= 0.1
kf.P *= 100

initial_detect = False

trajectory_arr = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for *xyxy, conf, cls in results.xyxy[0].tolist():
        # Detect ball only, ball id=32
        if cls == 32:
            detected_ball = True

            label = f'{results.names[int(cls)]}: {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            centroid_x = int((xyxy[0] + xyxy[2]) / 2)
            centroid_y = int((xyxy[1] + xyxy[3]) / 2)
            frame = cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

            # Use the ball's position as the beginning value of Kalman filter
            if initial_detect is False:
                initial_detect = True
                kf.x = np.array([centroid_x, centroid_y, 0, 0])
                trajectory_arr.append([centroid_x, centroid_y])

            kf.update(np.array([centroid_x, centroid_y]))

    if detected_ball:
        # Predict
        kf.predict()
        frame = cv2.circle(frame, (int(kf.x[0]), int(kf.x[1])), 5, (0, 0, 255), -1)
        trajectory_arr.append([int(kf.x[0]), int(kf.x[1])])

    # Draw trajectory
    for i in range(len(trajectory_arr)-1):
        frame = cv2.line(frame, (trajectory_arr[i][0], trajectory_arr[i][1]),
                         (trajectory_arr[i+1][0], trajectory_arr[i+1][1]), (0, 0, 0), 3)

    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
