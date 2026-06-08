import asyncio
import json
import cv2
import uvicorn
import base64
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from utils.video_processor import VideoProcessor
from utils.config import KNOWN_FACES_DIR
from utils.notifications import alert_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Driver Drowsiness & Identity Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_processor = VideoProcessor(str(KNOWN_FACES_DIR))


@app.get("/health")
def health():
    return {"status": "ok", "service": "driver-monitor", "version": "1.0.0"}


@app.post("/register-face/{name}")
async def register_face(name: str, file: UploadFile = File(...)):
    contents = await file.read()
    import numpy as np
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    success = video_processor.face_recognizer.register_face(name, img)
    if not success:
        raise HTTPException(400, "No face detected in image")
    return {"status": "registered", "name": name}


@app.get("/known-faces")
def get_known_faces():
    names = list(set(video_processor.face_recognizer.known_names))
    return {"faces": names, "count": len(names)}


@app.get("/alerts")
def get_alerts(limit: int = 20):
    return {"alerts": alert_system.get_recent_alerts(limit)}


@app.get("/status")
def get_status():
    return {
        "driver": {
            "name": video_processor.driver_name,
            "verified": video_processor.face_verified,
            "confidence": video_processor.face_confidence,
        },
        "emergency": video_processor.emergency_mode,
        "frame_count": video_processor.frame_count,
    }


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json({"error": "No camera available"})
            await websocket.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.005)
                continue

            jpg_bytes, telemetry = video_processor.process_frame(frame)
            b64_str = base64.b64encode(jpg_bytes).decode("utf-8")

            await websocket.send_json({
                "type": "frame",
                "data": b64_str,
                "telemetry": telemetry,
            })

            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if cap:
            cap.release()
        logger.info("Camera released")


@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Telemetry client connected")
    try:
        while True:
            await websocket.receive_text()
            status = {
                "driver": {
                    "name": video_processor.driver_name,
                    "verified": video_processor.face_verified,
                    "confidence": video_processor.face_confidence,
                },
                "status": {
                    "drowsy": video_processor.drowsiness_detector.drowsy,
                    "yawning": video_processor.yawning_detector.yawning,
                    "phone_in_use": video_processor.phone_detector.phone_in_use,
                    "emergency": video_processor.emergency_mode,
                },
                "metrics": {
                    "ear": video_processor.drowsiness_detector.ear_history[-1] if video_processor.drowsiness_detector.ear_history else 0,
                    "fps": video_processor.fps,
                },
                "frame_count": video_processor.frame_count,
            }
            await websocket.send_json(status)
    except WebSocketDisconnect:
        logger.info("Telemetry client disconnected")


if __name__ == "__main__":
    logger.info("Starting Driver Monitor Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
