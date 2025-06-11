from preprocessing.heart_beat.api_fetch import HeartRateMonitor  # your previous class file
from preprocessing.eye_pupil_dilation.pupil_detection_horizontal import PupilDiameterDetector
import cv2

class MentalHealthAnalyzer:
    def __init__(self, watch_mac):
        self.heart_monitor = HeartRateMonitor(watch_mac)
        self.pupil_detector = PupilDiameterDetector()
        self.camera = cv2.VideoCapture(0)

    def get_frame_data(self):
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        pupil_data = self.pupil_detector.process_frame(frame)
        return frame, pupil_data

    def get_combined_metrics(self):
        print("📸 Capturing frame...")
        frame, pupil_data = self.get_frame_data()
        if not pupil_data:
            return {"error": "Face or eyes not detected."}

        print("📟 Fetching BPM...")
        bpm = self.heart_monitor.read_heart_rate()

        combined_data = {
            "bpm": bpm,
            "left_pupil_diameter": pupil_data["left_pupil_diameter_px"],
            "right_pupil_diameter": pupil_data["right_pupil_diameter_px"],
            "eyes_closed": {
                "left": pupil_data["left_eye_closed"],
                "right": pupil_data["right_eye_closed"]
            },
            "feedback": pupil_data["feedback"]
        }
        return combined_data

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    analyzer = MentalHealthAnalyzer("XX:XX:XX:XX:XX:XX")  # Replace MAC
    result = analyzer.get_combined_metrics()
    print(result)
    analyzer.cleanup()
