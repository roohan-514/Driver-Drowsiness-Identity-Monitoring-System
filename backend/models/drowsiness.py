import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=0.25, consec_frames=48):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.ear_history = []
        self.eye_closed_counter = 0
        self.drowsy = False
        self.alerts_triggered = 0

    @staticmethod
    def _eye_aspect_ratio(landmarks, eye_indices):
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    def process_frame(self, landmarks_list):
        ear = 0.0
        face_detected = landmarks_list is not None

        if landmarks_list is not None:
            left_ear = self._eye_aspect_ratio(landmarks_list, self.LEFT_EYE)
            right_ear = self._eye_aspect_ratio(landmarks_list, self.RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)

            if ear < self.ear_threshold:
                self.eye_closed_counter += 1
            else:
                if self.eye_closed_counter >= self.consec_frames:
                    logger.info("Driver awake — eyes opened")
                self.eye_closed_counter = 0
                self.drowsy = False

            if self.eye_closed_counter >= self.consec_frames and not self.drowsy:
                self.drowsy = True
                self.alerts_triggered += 1
                logger.warning(f"DROWSINESS ALERT #{self.alerts_triggered}")

        return {
            "ear": round(ear, 3),
            "drowsy": self.drowsy,
            "eye_closed_counter": self.eye_closed_counter,
            "face_detected": face_detected,
            "alerts_triggered": self.alerts_triggered,
            "ear_history": self.ear_history[-10:] if self.ear_history else [],
        }
