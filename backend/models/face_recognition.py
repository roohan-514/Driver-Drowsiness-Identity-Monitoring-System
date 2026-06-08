import os
import pickle
import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (160, 160)


class FaceRecognizer:
    def __init__(self, known_faces_dir: str):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.known_encodings = []
        self.known_names = []
        self.unknown_threshold = 0.5
        self._load_known_faces()

    def _extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, FACE_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        emb = normalized.flatten()
        norm = np.linalg.norm(emb) + 1e-6
        return emb / norm

    def _load_known_faces(self):
        encodings_file = self.known_faces_dir / "encodings.pkl"
        if encodings_file.exists():
            with open(encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            logger.info(f"Loaded {len(self.known_encodings)} known face(s)")
        else:
            self._scan_known_faces()

    def _scan_known_faces(self):
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for img_path in self.known_faces_dir.glob(ext):
                name = img_path.stem
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) == 0:
                    continue
                x, y, w, h = faces[0]
                face_img = img[y:y + h, x:x + w]
                if face_img.size > 0:
                    emb = self._extract_embedding(face_img)
                    self.known_encodings.append(emb)
                    self.known_names.append(name)
                    logger.info(f"Registered face: {name}")
        self._save_encodings()

    def _save_encodings(self):
        encodings_file = self.known_faces_dir / "encodings.pkl"
        with open(encodings_file, "wb") as f:
            pickle.dump({
                "encodings": self.known_encodings,
                "names": self.known_names,
            }, f)

    def register_face(self, name: str, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            logger.warning(f"No face detected for registration: {name}")
            return False
        x, y, w, h = faces[0]
        face_img = image[y:y + h, x:x + w]
        if face_img.size == 0:
            return False
        emb = self._extract_embedding(face_img)
        self.known_encodings.append(emb)
        self.known_names.append(name)
        self._save_encodings()
        save_path = self.known_faces_dir / f"{name}.jpg"
        cv2.imwrite(str(save_path), image)
        logger.info(f"Registered new face: {name}")
        return True

    def recognize(self, face_img: np.ndarray) -> tuple:
        if face_img.size == 0:
            return "Unknown", 0.0
        emb = self._extract_embedding(face_img)
        if not self.known_encodings:
            return "Unknown", 0.0
        min_dist = float("inf")
        best_name = "Unknown"
        for known_emb, name in zip(self.known_encodings, self.known_names):
            dist = np.linalg.norm(emb - known_emb)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        confidence = max(0.0, 1.0 - min_dist)
        if min_dist > self.unknown_threshold:
            return "Unknown", confidence
        return best_name, confidence
