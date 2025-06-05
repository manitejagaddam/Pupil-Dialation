import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Extract eye landmarks using MediaPipe
def extract_pupil_series(video_path):
    cap = cv2.VideoCapture(video_path)
    pupil_sizes = []

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                # Use indices for left pupil
                # Sample: 468, 469, 470, 471 — approximate center
                cx = int(landmarks.landmark[468].x * frame.shape[1])
                cy = int(landmarks.landmark[468].y * frame.shape[0])
                pupil_sizes.append((cx, cy))  # You can expand to radius via contour

    cap.release()
    return np.array(pupil_sizes)

extract_pupil_series()
