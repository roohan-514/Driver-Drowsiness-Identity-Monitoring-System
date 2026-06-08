import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhoneUsageDetector:
    def __init__(self, confidence_threshold=0.25, consec_frames=8):
        self.confidence_threshold = confidence_threshold
        self.consec_frames = consec_frames
        self.phone_counter = 0
        self.phone_in_use = False
        self.total_phone_events = 0
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = Path(__file__).parent / "yolo11n.pt"
            if not model_path.exists():
                model_dir = Path(__file__).parent.parent
                model_path = model_dir / "yolo11n.pt"
            self.model = YOLO(str(model_path) if model_path.exists() else "yolo11n.pt")
            logger.info("YOLOv11 model loaded successfully")
        except ImportError:
            logger.warning("ultralytics not installed — phone detection disabled")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def process_frame(self, frame: np.ndarray):
        phone_detected = False
        detections = []
        confidence = 0.0

        if self.model is not None:
            results = self.model(frame, verbose=False, imgsz=320)[0]
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                cls = int(cls)
                if cls == 67 and conf >= self.confidence_threshold:
                    phone_detected = True
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 3),
                    })
                    confidence = max(confidence, conf)
            if not phone_detected and results.boxes.id is not None:
                for det in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    cls = int(cls)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = bbox_area / frame_area
                    if cls == 0 and area_ratio < 0.05 and conf >= self.confidence_threshold:
                        phone_detected = True
                        confidence = max(confidence, conf)
        else:
            phone_detected = self._fallback_hand_phone_detection(frame)

        if phone_detected:
            self.phone_counter += 1
        else:
            self.phone_counter = 0
            self.phone_in_use = False

        if self.phone_counter >= self.consec_frames and not self.phone_in_use:
            self.phone_in_use = True
            self.total_phone_events += 1
            logger.warning(f"PHONE USAGE DETECTED #{self.total_phone_events}")

        return {
            "phone_detected": phone_detected,
            "phone_in_use": self.phone_in_use,
            "phone_counter": self.phone_counter,
            "detections": detections,
            "confidence": round(confidence, 3),
            "total_phone_events": self.total_phone_events,
            "model_loaded": self.model is not None,
        }

    def _fallback_hand_phone_detection(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        phone_area_ratio = (np.sum(edges > 0) / (h * w))
        return phone_area_ratio > 0.15
