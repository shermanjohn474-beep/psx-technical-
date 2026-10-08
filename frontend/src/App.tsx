import { useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { setApiKey } from "./api";
import Dashboard from "./pages/Dashboard";
import ChartAnalysis from "./pages/ChartAnalysis";
import Screenshot from "./pages/Screenshot";
import Scanner from "./pages/Scanner";
import Backtest from "./pages/Backtest";
import Alerts from "./pages/Alerts";

export default function App() {
  const [key, setKey] = useState("");
  return (
    <div className="shell">
      <nav className="nav">
        <div className="logo">SHERMAN <span>SECURITIES</span></div>
        <div className="tag">PSX Technical Research Platform</div>
        <NavLink to="/" end>Dashboard</NavLink>
        <NavLink to="/chart/SYN-ASCT">Chart analysis</NavLink>
        <NavLink to="/screenshot">Screenshot analysis</NavLink>
        <NavLink to="/scanner">PSX scanner</NavLink>
        <NavLink to="/backtest">Backtesting</NavLink>
        <NavLink to="/alerts">Alerts</NavLink>
        <div style={{ marginTop: 24 }} className="no-print">
          <input type="password" placeholder="API key (if required)" value={key} onChange={(e) => setKey(e.target.value)}
            onBlur={() => key && setApiKey(key)} style={{ width: "100%", fontSize: 12 }} />
        </div>
        <p className="tag" style={{ marginTop: 18 }}>Research output only. Not investment advice. Scores are heuristics, not probabilities.</p>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chart/:symbol" element={<ChartAnalysis />} />
          <Route path="/screenshot" element={<Screenshot />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}
