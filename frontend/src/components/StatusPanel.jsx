import React from "react";

function StatusIndicator({ label, active, type = "warning", activeLabel = "DETECTED", inactiveLabel = "OK" }) {
  return (
    <div className={`status-indicator ${active ? `active ${type}` : "inactive"}`}>
      <span className={`indicator-dot ${active ? type : ""}`} />
      <span className="indicator-label">{label}</span>
      <span className="indicator-value">{active ? activeLabel : inactiveLabel}</span>
    </div>
  );
}

export default function StatusPanel({ telemetry }) {
  const s = telemetry?.status;
  const m = telemetry?.metrics;
  const d = telemetry?.driver;

  return (
    <div className="panel status-panel">
      <h3>Driver Status</h3>
      <div className="status-grid">
        <StatusIndicator
          label="Drowsiness"
          active={s?.drowsy}
          type="danger"
          activeLabel="ASLEEP"
          inactiveLabel="AWAKE"
        />
        <StatusIndicator
          label="Yawning"
          active={s?.yawning}
          type="warning"
        />
        <StatusIndicator
          label="Phone Usage"
          active={s?.phone_in_use}
          type="danger"
        />
        <StatusIndicator
          label="Face Detected"
          active={s?.face_detected}
          type="success"
          activeLabel="YES"
          inactiveLabel="NO"
        />
        <StatusIndicator
          label="Identity Verified"
          active={d?.verified}
          type="success"
          activeLabel="VERIFIED"
          inactiveLabel="NOT VERIFIED"
        />
        <StatusIndicator
          label="Looking Down"
          active={s?.looking_down}
          type="danger"
          activeLabel="ALERT"
          inactiveLabel="OK"
        />
        <StatusIndicator
          label="Looking Away"
          active={s?.looking_away}
          type="warning"
          activeLabel="ALERT"
          inactiveLabel="OK"
        />
        <StatusIndicator
          label="Face Lost"
          active={s?.face_lost_warning}
          type="warning"
          activeLabel="WARNING"
          inactiveLabel="OK"
        />
        <StatusIndicator
          label="Emergency"
          active={s?.emergency}
          type="danger"
        />
      </div>
      <div className="metrics">
        <div className="metric">
          <span className="metric-label">EAR</span>
          <span className={`metric-value ${m?.ear < 0.25 ? "danger" : ""}`}>
            {m?.ear?.toFixed(3) ?? "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">MAR</span>
          <span className={`metric-value ${m?.mar > 0.6 ? "warning" : ""}`}>
            {m?.mar?.toFixed(3) ?? "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Total Yawns</span>
          <span className="metric-value">{m?.total_yawns ?? 0}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Eye Counter</span>
          <span className="metric-value">{m?.eye_closed_counter ?? 0}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Phone Conf</span>
          <span className="metric-value">{(m?.phone_confidence * 100)?.toFixed(0) ?? "—"}%</span>
        </div>
        <div className="metric">
          <span className="metric-label">Face Conf</span>
          <span className="metric-value">
            {(d?.confidence * 100)?.toFixed(0) ?? "—"}%
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Gaze X</span>
          <span className="metric-value">{m?.nose_offset_x?.toFixed(3) ?? "—"}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Gaze Y</span>
          <span className="metric-value">{m?.nose_offset_y?.toFixed(3) ?? "—"}</span>
        </div>
      </div>
    </div>
  );
}
