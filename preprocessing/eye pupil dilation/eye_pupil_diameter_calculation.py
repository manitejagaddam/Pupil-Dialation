import cv2
import mediapipe as mp
import math

class PupilTracker:
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE_LIDS = [159, 145]  # top, bottom
    RIGHT_EYE_LIDS = [386, 374]  # top, bottom
    EYE_CLOSED_THRESH = 5.0  # Threshold to detect eye closure

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

    def euclidean_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_landmark_coords(self, landmarks, indices, w, h):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        landmarks = results.multi_face_landmarks[0].landmark

        # Get eye and iris landmarks
        left_iris_pts = self.get_landmark_coords(landmarks, self.LEFT_IRIS, w, h)
        right_iris_pts = self.get_landmark_coords(landmarks, self.RIGHT_IRIS, w, h)
        left_top, left_bottom = self.get_landmark_coords(landmarks, self.LEFT_EYE_LIDS, w, h)
        right_top, right_bottom = self.get_landmark_coords(landmarks, self.RIGHT_EYE_LIDS, w, h)

        left_open = self.euclidean_distance(left_top, left_bottom)
        right_open = self.euclidean_distance(right_top, right_bottom)

        left_closed = left_open < self.EYE_CLOSED_THRESH
        right_closed = right_open < self.EYE_CLOSED_THRESH

        # Draw eyelid landmarks
        for pt in [left_top, left_bottom, right_top, right_bottom]:
            cv2.circle(frame, pt, 2, (255, 0, 255), -1)

        if left_closed and right_closed:
            cv2.putText(frame, "Open BOTH eyes!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif left_closed:
            cv2.putText(frame, "Open LEFT eye!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif right_closed:
            cv2.putText(frame, "Open RIGHT eye!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            left_dia = self.euclidean_distance(left_iris_pts[0], left_iris_pts[2])
            right_dia = self.euclidean_distance(right_iris_pts[0], right_iris_pts[2])

            cv2.putText(frame, f"Left Pupil Dia: {left_dia:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Right Pupil Dia: {right_dia:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            for pt in left_iris_pts + right_iris_pts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        return frame


def main():
    tracker = PupilTracker()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = tracker.process_frame(frame)
        cv2.imshow("Pupil Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
