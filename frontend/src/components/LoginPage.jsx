import React, { useState } from "react";

export default function LoginPage({ onLogin }) {
  const [name, setName] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) {
      onLogin(name.trim());
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-icon">🚗</div>
        <h1>Driver Monitor</h1>
        <p className="login-subtitle">Driver Drowsiness &amp; Identity Monitoring System</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            className="login-input"
            placeholder="Enter driver name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
          <button type="submit" className="login-btn" disabled={!name.trim()}>
            Start Monitoring
          </button>
        </form>
      </div>
    </div>
  );
}
