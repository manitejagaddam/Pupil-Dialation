import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
import time

# === CONFIG ===
MODEL_PATH = "../models/cnn.h5"  # Path to your CNN model
WINDOW_SIZE = 30
HR_THRESHOLD = 75  # Adjust based on your data

# === Load model and scaler ===
print("[INFO] Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = MinMaxScaler(feature_range=(0, 1))  # normalize within window

# === Init webcam and MediaPipe ===
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

radius_window = deque(maxlen=WINDOW_SIZE)
time_start = time.time()

print("[INFO] Real-time HR Prediction Started. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    hr_text = "Estimating..."
    status_text = ""

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Pupil landmark indices
        pupil_indices = [468, 469, 470, 471, 472, 473]

        # Extract pupil coordinates
        coords = np.array([
            [int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])]
            for i in pupil_indices
        ])

        # Compute pupil center and radius
        center = np.mean(coords, axis=0)
        radius = np.mean(np.linalg.norm(coords - center, axis=1))
        radius_window.append(radius)
        print(f"[DEBUG] Pupil radius: {radius:.2f}")

        # Draw circle around pupil
        cv2.circle(frame, tuple(center.astype(int)), int(radius), (0, 255, 0), 2)

        # Predict once we have 30 frames of radius data
        if len(radius_window) == WINDOW_SIZE:
            # Normalize current window
            input_seq = np.array(radius_window).reshape(-1, 1)
            scaled = scaler.fit_transform(input_seq).reshape(1, WINDOW_SIZE, 1)

            # Predict heart rate
            prediction = model.predict(scaled, verbose=0)
            hr_pred = prediction[0][0]

            # Decide mental health status
            if hr_pred < HR_THRESHOLD:
                status_text = "Mental Health: LOW"
                color = (0, 0, 255)  # Red
            else:
                status_text = "Mental Health: NORMAL"
                color = (0, 255, 0)  # Green

            hr_text = f"HR: {hr_pred:.2f} BPM"

            # Debug info
            print(f"[DEBUG] Predicted HR: {hr_pred:.2f} | Status: {status_text}")

            # Display HR
            cv2.putText(frame, hr_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, status_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            cv2.putText(frame, "Collecting data...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # Show output
    cv2.imshow("Real-Time HR & Mental Health Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
