import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# define paths
base_dir = Path(r"C:\Users\raysh\OneDrive\Documents\bonsai\REAL\arucotracking")
video_path = base_dir / "unprocessed.avi"
output_path = base_dir / "processed.avi"
homography_path = base_dir / "homography.csv"

# load homography matrix
def load_homography(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return np.array([[float(x) for x in row] for row in reader])

H = load_homography(homography_path)

# aruco setup (with original-type markers)
aruco_id = 12
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()

# homography transform
def apply_homography(H, x, y):
    vector = np.array([x, y, 1.0])
    result = np.dot(H, vector)
    if result[2] != 0:
        return result[0] / result[2], result[1] / result[2]
    return float('nan'), float('nan')

# video setup
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
if not out.isOpened():
    raise IOError("Could not open VideoWriter")

timestamps, xs, ys, angles = [], [], [], []
frame_index = 0

# analysis
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # get timestamp
    time_sec = frame_index / fps
    frame_index += 1

    if (frame.shape[1], frame.shape[0]) != (width, height):
        frame = cv2.resize(frame, (width, height))

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == aruco_id:
                # position
                c = corners[i][0]
                center_x = float(np.mean(c[:, 0]))
                center_y = float(np.mean(c[:, 1]))
                transformed_x, transformed_y = apply_homography(H, center_x, center_y)

                # angle
                vector = c[1] - c[0]
                angle_deg = np.degrees(np.arctan2(vector[1], vector[0])) % 360

                timestamps.append(time_sec)
                xs.append(transformed_x)
                ys.append(transformed_y)
                angles.append(angle_deg)            

    out.write(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# finish & save to csv
cap.release()
out.release()
cv2.destroyAllWindows()

csv_output_path = base_dir / "offlineTrackingData.csv"

with open(csv_output_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x_position', 'y_position', 'angle_deg', 'time_sec']) # header
    writer.writerows(zip(xs, ys, angles, timestamps))

print(f"Tracking data saved to: {csv_output_path}")