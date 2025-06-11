import cv2
import mediapipe as mp
import math

# Landmarks for iris and eyelids
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_LIDS = [159, 145]   # top, bottom
RIGHT_EYE_LIDS = [386, 374]  # top, bottom

# Functions
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_center(pts):
    x = int(sum(p[0] for p in pts) / len(pts))
    y = int(sum(p[1] for p in pts) / len(pts))
    return (x, y)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye closed detection threshold
EYE_CLOSED_THRESH = 5.0

# Start capture
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

        # Get iris and eyelid points
        left_iris_pts = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in LEFT_IRIS]
        right_iris_pts = [(int(mesh[p].x * w), int(mesh[p].y * h)) for p in RIGHT_IRIS]

        left_top = (int(mesh[LEFT_EYE_LIDS[0]].x * w), int(mesh[LEFT_EYE_LIDS[0]].y * h))
        left_bottom = (int(mesh[LEFT_EYE_LIDS[1]].x * w), int(mesh[LEFT_EYE_LIDS[1]].y * h))
        right_top = (int(mesh[RIGHT_EYE_LIDS[0]].x * w), int(mesh[RIGHT_EYE_LIDS[0]].y * h))
        right_bottom = (int(mesh[RIGHT_EYE_LIDS[1]].x * w), int(mesh[RIGHT_EYE_LIDS[1]].y * h))

        # Calculate eye openness
        left_open = euclidean_distance(left_top, left_bottom)
        right_open = euclidean_distance(right_top, right_bottom)

        # Draw debug points
        for pt in [left_top, left_bottom, right_top, right_bottom]:
            cv2.circle(frame, pt, 2, (255, 0, 255), -1)

        # Determine closed eyes
        left_closed = left_open < EYE_CLOSED_THRESH
        right_closed = right_open < EYE_CLOSED_THRESH

        # Feedback logic
        if left_closed and right_closed:
            cv2.putText(frame, "Open BOTH eyes!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif left_closed:
            cv2.putText(frame, "Open your LEFT eye!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif right_closed:
            cv2.putText(frame, "Open your RIGHT eye!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            # Draw and calculate pupil features
            for p in left_iris_pts + right_iris_pts:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            left_center = get_center(left_iris_pts)
            right_center = get_center(right_iris_pts)
            cv2.circle(frame, left_center, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_center, 3, (255, 0, 0), -1)

            left_dia = euclidean_distance(left_iris_pts[0], left_iris_pts[2])
            right_dia = euclidean_distance(right_iris_pts[0], right_iris_pts[2])

            cv2.putText(frame, f"Left Dia: {left_dia:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Right Dia: {right_dia:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Eye Detection with Alerts", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
