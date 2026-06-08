import React from "react";

export default function VideoFeed({ frame, connected, telemetry, driverName }) {
  return (
    <div className="video-feed">
      <div className="video-header">
        <span className={`status-dot ${connected ? "connected" : "disconnected"}`} />
        <span>{connected ? "Camera Feed" : "Disconnected"}</span>
        {telemetry && (
          <span className="fps-badge">{telemetry.fps} FPS</span>
        )}
      </div>
      <div className="video-container">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Driver camera feed"
            className="video-frame"
          />
        ) : (
          <div className="video-placeholder">
            {connected ? "Waiting for video..." : "Connecting to server..."}
          </div>
        )}
      </div>
      {telemetry?.status?.emergency && (
        <div className="emergency-overlay">
          EMERGENCY — Driver Unresponsive
        </div>
      )}
      <div className="video-footer">
        <div className="video-meta">
          Frame: {telemetry?.frame ?? "-"} | 
          Login: {driverName ?? "-"} | 
          Face: {telemetry?.driver?.name ?? "Unknown"}
          {telemetry?.driver?.verified ? " ✓" : " ✗"}
        </div>
      </div>
    </div>
  );
}
