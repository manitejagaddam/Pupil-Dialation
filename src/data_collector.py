import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set up directories
output_dir = "../recordings"
video_path = os.path.join(output_dir, "eye_video.mp4")
csv_path = os.path.join(output_dir, "pupil_hr_log.csv")
os.makedirs(output_dir, exist_ok=True)

# Parameters
duration = 60  # seconds
fps = 20
frame_width = 640
frame_height = 480

# Initialize video capture and writer
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

pupil_log = []
frame_idx = 0
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Approximate left pupil center from mesh indices 468–473
        pupil_indices = [468, 469, 470, 471, 472, 473]
        pupil_coords = np.array([
            [int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])]
            for i in pupil_indices
        ])
        center = np.mean(pupil_coords, axis=0)
        radius = np.mean(np.linalg.norm(pupil_coords - center, axis=1))

        # Save data
        elapsed = (datetime.now() - start_time).total_seconds()
        heartbeat = 75 + 5 * np.sin(elapsed / 10)  # Simulated HR
        pupil_log.append({
            "timestamp": datetime.now().isoformat(),
            "frame_idx": frame_idx,
            "elapsed_sec": round(elapsed, 2),
            "pupil_x": round(center[0], 2),
            "pupil_y": round(center[1], 2),
            "pupil_radius": round(radius, 2),
            "heart_rate": round(heartbeat, 2)
        })

        # Draw for preview (optional)
        cv2.circle(frame, tuple(center.astype(int)), int(radius), (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1

    if elapsed >= duration:
        break

# Clean up
cap.release()
out.release()
face_mesh.close()

# Save CSV
df = pd.DataFrame(pupil_log)
df.to_csv(csv_path, index=False)

video_path, csv_path
