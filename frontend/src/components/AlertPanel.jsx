import React from "react";

function AlertItem({ alert, index }) {
  const severityClass = alert.severity || "info";
  return (
    <div className={`alert-item ${severityClass}`} key={index}>
      <span className="alert-time">{alert.formatted_time}</span>
      <span className="alert-type">{alert.type}</span>
      <span className="alert-message">{alert.message}</span>
    </div>
  );
}

export default function AlertPanel({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="panel alert-panel">
        <h3>Recent Alerts</h3>
        <div className="alert-empty">No alerts — system nominal</div>
      </div>
    );
  }

  return (
    <div className="panel alert-panel">
      <h3>Recent Alerts</h3>
      <div className="alert-list">
        {alerts
          .slice()
          .reverse()
          .map((alert, i) => (
            <AlertItem alert={alert} key={i} />
          ))}
      </div>
    </div>
  );
}
