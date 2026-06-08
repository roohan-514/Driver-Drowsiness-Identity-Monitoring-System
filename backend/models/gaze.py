import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GazeDetector:
    def __init__(self, down_threshold=0.15, away_threshold=0.18, consec_frames=30):
        self.down_threshold = down_threshold
        self.away_threshold = away_threshold
        self.consec_frames = consec_frames
        self.looking_down_counter = 0
        self.looking_away_counter = 0
        self.looking_down = False
        self.looking_away = False
        self.total_distractions = 0

    def process_frame(self, landmarks_list, img_w, img_h):
        looking_down = False
        looking_away = False
        nose_offset_x = 0.0
        nose_offset_y = 0.0

        if landmarks_list is not None and len(landmarks_list) > 470:
            nose_tip = landmarks_list[1]
            left_eye = np.array([landmarks_list[33].x, landmarks_list[33].y])
            right_eye = np.array([landmarks_list[263].x, landmarks_list[263].y])
            face_center = (left_eye + right_eye) / 2.0
            nose_x, nose_y = nose_tip.x, nose_tip.y
            nose_offset_x = nose_x - face_center[0]
            nose_offset_y = nose_y - face_center[1]

            left_ear_dist = np.linalg.norm(nose_x - left_eye[0])
            right_ear_dist = np.linalg.norm(nose_x - right_eye[0])
            if left_ear_dist + right_ear_dist > 0:
                head_turn_ratio = (left_ear_dist - right_ear_dist) / (left_ear_dist + right_ear_dist)
            else:
                head_turn_ratio = 0.0

            if nose_offset_y > self.down_threshold:
                looking_down = True

            if abs(head_turn_ratio) > 0.3 or abs(nose_offset_x) > self.away_threshold:
                looking_away = True

        if looking_down:
            self.looking_down_counter += 1
        else:
            if self.looking_down_counter >= self.consec_frames:
                logger.info("Driver looking forward again")
            self.looking_down_counter = 0
            self.looking_down = False

        if looking_away:
            self.looking_away_counter += 1
        else:
            if self.looking_away_counter >= self.consec_frames:
                logger.info("Driver facing forward again")
            self.looking_away_counter = 0
            self.looking_away = False

        if self.looking_down_counter >= self.consec_frames and not self.looking_down:
            self.looking_down = True
            self.total_distractions += 1
            logger.warning(f"LOOKING DOWN #{self.total_distractions}")

        if self.looking_away_counter >= self.consec_frames and not self.looking_away:
            self.looking_away = True
            self.total_distractions += 1
            logger.warning(f"LOOKING AWAY #{self.total_distractions}")

        return {
            "looking_down": self.looking_down,
            "looking_away": self.looking_away,
            "looking_down_counter": self.looking_down_counter,
            "looking_away_counter": self.looking_away_counter,
            "nose_offset_x": round(nose_offset_x, 4),
            "nose_offset_y": round(nose_offset_y, 4),
            "total_distractions": self.total_distractions,
        }
