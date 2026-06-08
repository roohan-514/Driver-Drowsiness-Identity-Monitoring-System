import time
import cv2
import numpy as np
import logging
import threading
from pathlib import Path

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.config import (
    EYE_AR_THRESHOLD, EYE_AR_CONSEC_FRAMES,
    YAWN_AR_THRESHOLD, YAWN_CONSEC_FRAMES,
    PHONE_CONFIDENCE_THRESHOLD, PHONE_CONSEC_FRAMES,
    GAZE_DOWN_THRESHOLD, GAZE_AWAY_THRESHOLD, GAZE_CONSEC_FRAMES,
    FACE_LOST_WARNING_FRAMES, FACE_LOST_CRITICAL_FRAMES,
    FRAME_WIDTH, FRAME_HEIGHT,
)
from models.face_recognition import FaceRecognizer
from models.drowsiness import DrowsinessDetector
from models.yawning import YawningDetector
from models.phone_usage import PhoneUsageDetector
from models.gaze import GazeDetector
from utils.notifications import alert_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANDMARKER_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
FACE_INTERVAL = 3
PHONE_INTERVAL = 5


def _play_alarm():
    if HAS_SOUND:
        for _ in range(10):
            winsound.Beep(1000, 400)
            time.sleep(0.05)
            winsound.Beep(800, 400)
            time.sleep(0.05)
            winsound.Beep(1200, 400)
            time.sleep(0.05)


class VideoProcessor:
    def __init__(self, known_faces_dir: str):
        self.face_recognizer = FaceRecognizer(known_faces_dir)
        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=EYE_AR_THRESHOLD,
            consec_frames=EYE_AR_CONSEC_FRAMES,
        )
        self.yawning_detector = YawningDetector(
            mar_threshold=YAWN_AR_THRESHOLD,
            consec_frames=YAWN_CONSEC_FRAMES,
        )
        self.phone_detector = PhoneUsageDetector(
            confidence_threshold=PHONE_CONFIDENCE_THRESHOLD,
            consec_frames=PHONE_CONSEC_FRAMES,
        )
        self.gaze_detector = GazeDetector(
            down_threshold=GAZE_DOWN_THRESHOLD,
            away_threshold=GAZE_AWAY_THRESHOLD,
            consec_frames=GAZE_CONSEC_FRAMES,
        )
        self.face_landmarker = None
        self._init_landmarker()
        self.frame_count = 0
        self.fps = 0
        self._fps_start = time.time()
        self._fps_frames = 0
        self.driver_name = "Unknown"
        self.face_confidence = 0.0
        self.face_verified = False
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        self.emergency_mode = False
        self.unresponsive_frames = 0
        self.max_unresponsive_frames = 150
        self.face_lost_frames = 0
        self.face_lost_warning = False
        self._alarm_playing = False

        self._last_landmarks = None
        self._last_face_bbox = None
        self._face_miss_count = 0

    def _init_landmarker(self):
        if not LANDMARKER_MODEL.exists():
            logger.warning(f"Face landmarker model not found at {LANDMARKER_MODEL}")
            return
        try:
            base = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker initialized")
        except Exception as e:
            logger.error(f"Failed to init FaceLandmarker: {e}")

    @staticmethod
    def _get_face_bbox(landmarks, img_w, img_h):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = int(max(0, min(xs)) * img_w)
        y1 = int(max(0, min(ys)) * img_h)
        x2 = int(min(img_w, max(xs)) * img_w)
        y2 = int(min(img_h, max(ys)) * img_h)
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_w, x2 + margin_x)
        y2 = min(img_h, y2 + margin_y)
        return x1, y1, x2, y2

    def process_frame(self, frame: np.ndarray):
        self.frame_count += 1
        self._fps_frames += 1
        if time.time() - self._fps_start >= 1.0:
            self.fps = self._fps_frames
            self._fps_frames = 0
            self._fps_start = time.time()

        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        h, w = frame_resized.shape[:2]
        do_face = self.frame_count % FACE_INTERVAL == 0
        do_phone = self.frame_count % PHONE_INTERVAL == 0
        landmarks_list = self._last_landmarks
        face_bbox = self._last_face_bbox
        face_lost_now = False

        if do_face and self.face_landmarker is not None:
            try:
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                result = self.face_landmarker.detect(mp_image)
                if result.face_landmarks:
                    landmarks_list = result.face_landmarks[0]
                    x1, y1, x2, y2 = self._get_face_bbox(landmarks_list, w, h)
                    face_bbox = (x1, y1, x2, y2)
                    self._last_landmarks = landmarks_list
                    self._last_face_bbox = face_bbox
                    self._face_miss_count = 0
                    face_lost_now = False
                else:
                    face_lost_now = True
                    self._face_miss_count += 1
                    if self._face_miss_count >= 3:
                        self._last_landmarks = None
                        self._last_face_bbox = None
                        landmarks_list = None
                        face_bbox = None
            except Exception as e:
                logger.error(f"Landmarker error: {e}")

        drowsy_data = self.drowsiness_detector.process_frame(landmarks_list)
        yawn_data = self.yawning_detector.process_frame(landmarks_list)
        gaze_data = self.gaze_detector.process_frame(landmarks_list, w, h)

        if do_phone:
            phone_data = self.phone_detector.process_frame(frame_resized)
        else:
            phone_data = {
                "phone_detected": False,
                "phone_in_use": self.phone_detector.phone_in_use,
                "phone_counter": self.phone_detector.phone_counter,
                "detections": [],
                "confidence": 0.0,
                "total_phone_events": self.phone_detector.total_phone_events,
                "model_loaded": self.phone_detector.model is not None,
            }

        face_detected = landmarks_list is not None

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_img = frame_resized[y1:y2, x1:x2]
            if face_img.size > 0:
                name, conf = self.face_recognizer.recognize(face_img)
                self.driver_name = name
                self.face_confidence = round(conf, 3)
                self.face_verified = name != "Unknown" and conf > 0.5

        if not face_detected:
            self.driver_name = "Unknown"
            self.face_confidence = 0.0
            self.face_verified = False

        now = time.time()

        if drowsy_data["drowsy"] and now - self.last_alert_time > self.alert_cooldown:
            alert_system.trigger_alert("drowsiness", f"DRIVER ASLEEP! WAKE UP! EAR: {drowsy_data['ear']:.3f}", "critical")
            self.last_alert_time = now
            if not self._alarm_playing:
                self._alarm_playing = True
                threading.Thread(target=self._play_alarm_sound, daemon=True).start()
                logger.warning("ALARM SOUND PLAYING — WAKE UP DRIVER!")

        if yawn_data["yawning"] and now - self.last_alert_time > self.alert_cooldown:
            alert_system.trigger_alert("yawning", f"Yawn detected! MAR: {yawn_data['mar']:.3f}", "warning")
            self.last_alert_time = now

        if phone_data["phone_in_use"] and now - self.last_alert_time > self.alert_cooldown:
            alert_system.trigger_alert("phone_usage", "WARNING: Phone in use while driving!", "critical")
            self.last_alert_time = now

        if gaze_data["looking_down"] and now - self.last_alert_time > self.alert_cooldown:
            alert_system.trigger_alert("looking_down", "LOOK AT THE ROAD! Driver looking down!", "critical")
            self.last_alert_time = now

        if gaze_data["looking_away"] and now - self.last_alert_time > self.alert_cooldown:
            alert_system.trigger_alert("looking_away", "Driver not facing forward! Pay attention!", "warning")
            self.last_alert_time = now

        if not face_detected:
            self.face_lost_frames += 1
            if self.face_lost_frames == FACE_LOST_WARNING_FRAMES and not self.face_lost_warning:
                self.face_lost_warning = True
                alert_system.trigger_alert("face_lost", "No face detected — driver not in position!", "warning")
                self.last_alert_time = now
            if self.face_lost_frames >= FACE_LOST_CRITICAL_FRAMES and now - self.last_alert_time > self.alert_cooldown:
                alert_system.trigger_alert("face_lost", "Face not visible — check on driver!", "critical")
                self.last_alert_time = now
        else:
            self.face_lost_frames = 0
            self.face_lost_warning = False

        if not face_detected:
            self.unresponsive_frames += 1
        else:
            self.unresponsive_frames = 0
            self.emergency_mode = False

        if self.unresponsive_frames >= self.max_unresponsive_frames and not self.emergency_mode:
            self.emergency_mode = True
            alert_system.trigger_alert("emergency", "EMERGENCY: Driver unresponsive! Activating emergency protocol!", "critical")
            alert_system.send_sms("EMERGENCY: Driver unresponsive. Location required.")

        overlay = frame_resized.copy()

        text_x_right = w - 300
        text_y = 40
        line_h = 30

        if face_detected:
            cv2.putText(overlay, f"Driver: {self.driver_name}", (text_x_right, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += line_h
            cv2.putText(overlay, f"EAR: {drowsy_data['ear']:.3f}", (text_x_right, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            text_y += line_h
            cv2.putText(overlay, f"MAR: {yawn_data['mar']:.3f}", (text_x_right, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            text_y += line_h
            cv2.putText(overlay, f"Yawns: {yawn_data['total_yawns']}", (text_x_right, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(overlay, "NO FACE DETECTED", (text_x_right, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            text_y += line_h
            if self.face_lost_warning:
                cv2.putText(overlay, "MOVE INTO CAMERA VIEW", (text_x_right, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        text_y = h - 30
        cv2.putText(overlay, f"FPS: {self.fps}", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if drowsy_data["drowsy"]:
            cv2.putText(overlay, "DRIVER ASLEEP!", (w // 2 - 130, h // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
            cv2.putText(overlay, "WAKE UP! WAKE UP!", (w // 2 - 140, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(overlay, "!! ALARM !!", (w // 2 - 80, h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            overlay = cv2.addWeighted(overlay, 1.0, np.full_like(overlay, (0, 0, 40)), 0.2, 0)

        if phone_data["phone_in_use"]:
            cv2.putText(overlay, "PHONE IN USE", (w // 2 - 90, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(overlay, "PUT PHONE AWAY!", (w // 2 - 100, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if yawn_data["yawning"]:
            cv2.putText(overlay, "YAWNING", (w - 250, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            overlay = cv2.addWeighted(overlay, 1.0, np.full_like(overlay, (0, 30, 30)), 0.1, 0)

        if gaze_data["looking_down"]:
            cv2.putText(overlay, "LOOK AT THE ROAD!", (w // 2 - 120, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(overlay, "STOP LOOKING DOWN", (w // 2 - 110, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if gaze_data["looking_away"] and not gaze_data["looking_down"]:
            cv2.putText(overlay, "FACE THE ROAD", (w // 2 - 90, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        if self.emergency_mode:
            overlay = cv2.addWeighted(overlay, 0.5, np.full_like(overlay, (0, 0, 255)), 0.5, 0)
            cv2.putText(overlay, "EMERGENCY", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            cv2.putText(overlay, "SOS SENT", (w // 2 - 70, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        if self.face_lost_warning and not self.emergency_mode:
            cv2.putText(overlay, "NO FACE", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 60])
        jpg_bytes = buffer.tobytes()

        telemetry = {
            "frame": self.frame_count,
            "fps": self.fps,
            "driver": {
                "name": self.driver_name,
                "verified": self.face_verified,
                "confidence": self.face_confidence,
            },
            "status": {
                "drowsy": drowsy_data["drowsy"],
                "yawning": yawn_data["yawning"],
                "phone_in_use": phone_data["phone_in_use"],
                "face_detected": face_detected,
                "face_lost_warning": self.face_lost_warning,
                "looking_down": gaze_data["looking_down"],
                "looking_away": gaze_data["looking_away"],
                "emergency": self.emergency_mode,
                "unresponsive_frames": self.unresponsive_frames,
            },
            "metrics": {
                "ear": drowsy_data["ear"],
                "mar": yawn_data["mar"],
                "eye_closed_counter": drowsy_data["eye_closed_counter"],
                "phone_confidence": phone_data["confidence"],
                "total_yawns": yawn_data["total_yawns"],
                "nose_offset_x": gaze_data["nose_offset_x"],
                "nose_offset_y": gaze_data["nose_offset_y"],
            },
            "alerts": alert_system.get_recent_alerts(5),
        }

        return jpg_bytes, telemetry

    def _play_alarm_sound(self):
        try:
            _play_alarm()
        finally:
            self._alarm_playing = False
