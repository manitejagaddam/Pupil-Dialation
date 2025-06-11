import cv2
import mediapipe as mp
import numpy as np
import math

# Eye/Iris landmark indices from MediaPipe FaceMesh
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Get pupil center as mean of 4 iris points
def get_center(pts):
    x = int(sum(p[0] for p in pts) / len(pts))
    y = int(sum(p[1] for p in pts) / len(pts))
    return (x, y)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        # Get left and right iris points
        left_iris_pts = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in LEFT_IRIS]
        right_iris_pts = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in RIGHT_IRIS]

        # Draw iris landmarks
        for p in left_iris_pts + right_iris_pts:
            cv2.circle(frame, p, 2, (0, 255, 0), -1)

        # Calculate pupil center
        left_center = get_center(left_iris_pts)
        right_center = get_center(right_iris_pts)

        # Draw centers
        cv2.circle(frame, left_center, 3, (255, 0, 0), -1)
        cv2.circle(frame, right_center, 3, (255, 0, 0), -1)

        # Estimate pupil diameter (horizontal only for now)
        left_diameter = euclidean_distance(left_iris_pts[0], left_iris_pts[2])
        right_diameter = euclidean_distance(right_iris_pts[0], right_iris_pts[2])

        # Print to frame
        cv2.putText(frame, f"Left Dia: {left_diameter:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Right Dia: {right_diameter:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Pupil Feature Extractor", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
