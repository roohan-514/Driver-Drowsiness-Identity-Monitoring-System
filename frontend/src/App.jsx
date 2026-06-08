import React, { useState } from "react";
import Dashboard from "./components/Dashboard";
import LoginPage from "./components/LoginPage";

export default function App() {
  const [driverName, setDriverName] = useState(null);

  if (!driverName) {
    return <LoginPage onLogin={setDriverName} />;
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>Driver Drowsiness &amp; Identity Monitor</h1>
          <span className="badge badge-live">LIVE</span>
        </div>
        <div className="header-right">
          <span className="driver-badge">Driver: {driverName}</span>
          <button className="logout-btn" onClick={() => setDriverName(null)}>
            Change Driver
          </button>
        </div>
      </header>
      <main>
        <Dashboard driverName={driverName} />
      </main>
    </div>
  );
}
