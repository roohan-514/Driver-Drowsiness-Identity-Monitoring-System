import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YawningDetector:
    def __init__(self, mar_threshold=0.6, consec_frames=30):
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.mar_history = []
        self.yawn_counter = 0
        self.yawning = False
        self.total_yawns = 0

    @staticmethod
    def _mouth_aspect_ratio(landmarks):
        upper_lip = np.array([landmarks[13].x, landmarks[13].y])
        lower_lip = np.array([landmarks[14].x, landmarks[14].y])
        left_mouth = np.array([landmarks[61].x, landmarks[61].y])
        right_mouth = np.array([landmarks[291].x, landmarks[291].y])
        vertical = np.linalg.norm(upper_lip - lower_lip)
        horizontal = np.linalg.norm(left_mouth - right_mouth)
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def process_frame(self, landmarks_list):
        mar = 0.0
        face_detected = landmarks_list is not None

        if landmarks_list is not None:
            mar = self._mouth_aspect_ratio(landmarks_list)
            self.mar_history.append(mar)
            if len(self.mar_history) > 30:
                self.mar_history.pop(0)

            if mar > self.mar_threshold:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0
                self.yawning = False

            if self.yawn_counter >= self.consec_frames and not self.yawning:
                self.yawning = True
                self.total_yawns += 1
                logger.warning(f"YAWN DETECTED #{self.total_yawns}")

        return {
            "mar": round(mar, 3),
            "yawning": self.yawning,
            "yawn_counter": self.yawn_counter,
            "total_yawns": self.total_yawns,
            "face_detected": face_detected,
        }
