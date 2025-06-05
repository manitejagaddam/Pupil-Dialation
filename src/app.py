import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
import time

# === CONFIG ===
MODEL_PATH = "models\cnn.h5"  # replace with your model path
WINDOW_SIZE = 30

# === Load model and scaler ===
model = tf.keras.models.load_model(MODEL_PATH)
scaler = MinMaxScaler(feature_range=(0, 1))  # We'll fit this live on 30-frame windows

# === Init webcam and MediaPipe ===
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

radius_window = deque(maxlen=WINDOW_SIZE)
time_start = time.time()

print("[INFO] Real-time prediction started. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    hr_text = "Estimating..."

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        pupil_indices = [468, 469, 470, 471, 472, 473]

        coords = np.array([
            [int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])]
            for i in pupil_indices
        ])

        center = np.mean(coords, axis=0)
        radius = np.mean(np.linalg.norm(coords - center, axis=1))
        radius_window.append(radius)

        # Draw eye circle
        cv2.circle(frame, tuple(center.astype(int)), int(radius), (0, 255, 0), 2)

        if len(radius_window) == WINDOW_SIZE:
            # Normalize live window
            input_seq = np.array(radius_window).reshape(-1, 1)
            scaled = scaler.fit_transform(input_seq).reshape(1, WINDOW_SIZE, 1)

            prediction = model.predict(scaled, verbose=0)
            hr_pred = prediction[0][0]
            hr_text = f"HR: {hr_pred:.2f} BPM"

    # Show frame
    cv2.putText(frame, hr_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Real-Time HR Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
