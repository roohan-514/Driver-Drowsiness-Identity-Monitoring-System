import React from "react";
import { useVideoStream } from "../hooks/useWebSocket";
import VideoFeed from "./VideoFeed";
import StatusPanel from "./StatusPanel";
import AlertPanel from "./AlertPanel";
import MetricsChart from "./MetricsChart";

const WS_URL = "ws://localhost:8000/ws/video";

export default function Dashboard({ driverName }) {
  const { connected, frame, telemetry } = useVideoStream(WS_URL);

  return (
    <div className="dashboard">
      <div className="dashboard-main">
        <div className="video-section">
          <VideoFeed
            frame={frame}
            connected={connected}
            telemetry={telemetry}
            driverName={driverName}
          />
        </div>
        <div className="sidebar">
          <StatusPanel telemetry={telemetry} />
          <MetricsChart telemetry={telemetry} />
        </div>
      </div>
      <div className="dashboard-bottom">
        <AlertPanel alerts={telemetry?.alerts} />
      </div>
    </div>
  );
}
