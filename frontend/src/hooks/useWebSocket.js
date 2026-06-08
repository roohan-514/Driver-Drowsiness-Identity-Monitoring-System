import { useEffect, useRef, useState, useCallback } from "react";

export function useVideoStream(url) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const reconnectTimeout = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "frame") {
          setFrame(msg.data);
          setTelemetry(msg.telemetry);
        }
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimeout.current = setTimeout(() => {
        connect();
      }, 2000);
    };

    ws.onerror = (err) => {
      console.error("WS error:", err);
      ws.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { connected, frame, telemetry };
}
