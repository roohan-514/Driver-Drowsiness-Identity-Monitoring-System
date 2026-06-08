import React, { useRef, useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function MetricsChart({ telemetry }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    if (!telemetry) return;
    const point = {
      time: new Date().toLocaleTimeString(),
      ear: telemetry.metrics?.ear ?? 0,
      mar: telemetry.metrics?.mar ?? 0,
    };
    setData((prev) => {
      const next = [...prev, point];
      if (next.length > 60) return next.slice(-60);
      return next;
    });
  }, [telemetry]);

  return (
    <div className="panel chart-panel">
      <h3>EAR / MAR Trend (last 60 frames)</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "#888" }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: "#888" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1a1a2e",
              border: "1px solid #333",
              borderRadius: "8px",
              fontSize: 12,
            }}
          />
          <Line
            type="monotone"
            dataKey="ear"
            stroke="#00d4ff"
            dot={false}
            strokeWidth={2}
            name="EAR"
          />
          <Line
            type="monotone"
            dataKey="mar"
            stroke="#ff6b6b"
            dot={false}
            strokeWidth={2}
            name="MAR"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
