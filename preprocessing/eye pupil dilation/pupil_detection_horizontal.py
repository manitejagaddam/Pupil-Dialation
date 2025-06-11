import cv2
import mediapipe as mp
import math

class PupilDiameterDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # to get iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils

        self.RIGHT_IRIS_IDS = [469, 471]  
        self.LEFT_IRIS_IDS = [474, 476]

        # Eye landmark indices for eye aspect ratio (EAR) to detect closed eyes
        self.RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_IDS = [362, 385, 387, 263, 373, 380]

        self.EAR_THRESHOLD = 0.15

    def _landmark_to_pixel(self, landmark, image_width, image_height):
        return int(landmark.x * image_width), int(landmark.y * image_height)

    def _euclidean_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _calculate_ear(self, landmarks, eye_indices, image_w, image_h):
        # EAR formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        p1 = self._landmark_to_pixel(landmarks[eye_indices[0]], image_w, image_h)
        p2 = self._landmark_to_pixel(landmarks[eye_indices[1]], image_w, image_h)
        p3 = self._landmark_to_pixel(landmarks[eye_indices[2]], image_w, image_h)
        p4 = self._landmark_to_pixel(landmarks[eye_indices[3]], image_w, image_h)
        p5 = self._landmark_to_pixel(landmarks[eye_indices[4]], image_w, image_h)
        p6 = self._landmark_to_pixel(landmarks[eye_indices[5]], image_w, image_h)

        vertical1 = self._euclidean_distance(p2, p6)
        vertical2 = self._euclidean_distance(p3, p5)
        horizontal = self._euclidean_distance(p1, p4)

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def _eye_closed(self, landmarks, eye_indices, w, h):
        ear = self._calculate_ear(landmarks, eye_indices, w, h)
        return ear < self.EAR_THRESHOLD

    def get_pupil_diameter(self, landmarks, iris_indices, image_w, image_h):
        left_edge = self._landmark_to_pixel(landmarks[iris_indices[0]], image_w, image_h)
        right_edge = self._landmark_to_pixel(landmarks[iris_indices[1]], image_w, image_h)
        diameter = self._euclidean_distance(left_edge, right_edge)
        return diameter

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None  # No face detected

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Calculate pupil diameters
        left_pupil_diameter = self.get_pupil_diameter(face_landmarks, self.LEFT_IRIS_IDS, w, h)
        right_pupil_diameter = self.get_pupil_diameter(face_landmarks, self.RIGHT_IRIS_IDS, w, h)

        # Check if eyes are closed
        left_eye_closed = self._eye_closed(face_landmarks, self.LEFT_EYE_IDS, w, h)
        right_eye_closed = self._eye_closed(face_landmarks, self.RIGHT_EYE_IDS, w, h)

        # Prepare feedback
        feedback = []
        if left_eye_closed:
            feedback.append("Please open your LEFT eye")
        if right_eye_closed:
            feedback.append("Please open your RIGHT eye")

        return {
            "left_pupil_diameter_px": left_pupil_diameter,
            "right_pupil_diameter_px": right_pupil_diameter,
            "left_eye_closed": left_eye_closed,
            "right_eye_closed": right_eye_closed,
            "feedback": feedback
        }

# Usage example
if __name__ == "__main__":
    detector = PupilDiameterDetector()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        if result:
            # Show pupil diameter on frame
            cv2.putText(frame, f"Left Pupil: {result['left_pupil_diameter_px']:.1f}px", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Pupil: {result['right_pupil_diameter_px']:.1f}px", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show eye closed feedback
            y0 = 100
            for i, msg in enumerate(result['feedback']):
                cv2.putText(frame, msg, (30, y0 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Pupil Diameter & Eye Status", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()
