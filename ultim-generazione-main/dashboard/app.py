"""
Local control-room dashboard (Phase 9).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List

from flask import Flask, jsonify, render_template_string

logger = logging.getLogger("Dashboard")


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sala di controllo V18</title>
  <style>
    :root { --bg:#0b0f14; --card:#151c24; --muted:#93a4b7; --text:#e8edf3; --ok:#26d07c; --bad:#ff5d73; --accent:#5fa8ff; }
    body { margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap { max-width:1200px; margin:24px auto; padding:0 16px; }
    h1 { font-size:22px; margin:0 0 14px; }
    .grid { display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:10px; margin-bottom:12px; }
    .card { background:var(--card); border:1px solid #202a35; border-radius:10px; padding:10px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; }
    .value { font-size:18px; font-weight:700; margin-top:4px; }
    .status-ok { color:var(--ok); } .status-bad { color:var(--bad); }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid #27313d; padding:8px; text-align:left; }
    th { color:var(--muted); font-weight:600; }
    .pill { background:#1e2c3d; color:var(--accent); border-radius:12px; padding:2px 8px; font-size:12px; }
    #logs { max-height:220px; overflow:auto; font-family:ui-monospace,Consolas,monospace; font-size:12px; white-space:pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>🧠 Sala di controllo V18</h1>

    <div class="grid">
      <div class="card"><div class="label">System</div><div class="value" id="systemStatus">-</div></div>
      <div class="card"><div class="label">Mode</div><div class="value" id="paperMode">-</div></div>
      <div class="card"><div class="label">Balance</div><div class="value" id="balance">-</div></div>
      <div class="card"><div class="label">Global Win Rate</div><div class="value" id="winRate">-</div></div>
      <div class="card"><div class="label">Total PnL</div><div class="value" id="totalPnl">-</div></div>
    </div>

    <div class="card" style="margin-bottom:12px;">
      <div class="label" style="margin-bottom:8px;">Active Positions</div>
      <table>
        <thead>
          <tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>Current</th><th>PnL</th><th>Strategy</th></tr>
        </thead>
        <tbody id="positionsBody"></tbody>
      </table>
    </div>

    <div class="card" style="margin-bottom:12px;">
      <div class="label" style="margin-bottom:8px;">MetaAgent Weights</div>
      <div id="weights"></div>
    </div>

    <div class="card">
      <div class="label" style="margin-bottom:8px;">Recent Signals / Logs</div>
      <div id="logs"></div>
    </div>
  </div>

  <script>
    const fmtPct = (v) => `${(Number(v || 0) * 100).toFixed(2)}%`;
    const fmtNum = (v, n=4) => Number(v || 0).toFixed(n);
    const fmtPnl = (v) => `${v >= 0 ? '+' : ''}${Number(v || 0).toFixed(4)}`;

    function renderState(data) {
      const systemEl = document.getElementById('systemStatus');
      const running = !!data.system_running;
      systemEl.textContent = running ? 'RUNNING' : 'STOPPED';
      systemEl.className = `value ${running ? 'status-ok' : 'status-bad'}`;
      document.getElementById('paperMode').textContent = data.paper_trading ? 'PAPER' : 'LIVE';
      document.getElementById('balance').textContent = fmtNum(data.balance, 2);
      document.getElementById('winRate').textContent = fmtPct(data.global_win_rate);
      const totalPnlEl = document.getElementById('totalPnl');
      totalPnlEl.textContent = `${fmtPnl(data.total_pnl)} (${Number(data.pnl_pct || 0).toFixed(2)}%)`;
      totalPnlEl.className = `value ${Number(data.total_pnl || 0) >= 0 ? 'status-ok' : 'status-bad'}`;

      const agentOrder = ['pattern', 'regime', 'confluence', 'risk', 'sentiment'];
      const chunks = agentOrder.map((k) => {
        const v = data.agent_weights && Object.prototype.hasOwnProperty.call(data.agent_weights, k)
          ? data.agent_weights[k]
          : null;
        return `<span class="pill">${k}: ${v === null ? 'n/a' : Number(v).toFixed(3)}</span>`;
      });
      document.getElementById('weights').innerHTML = chunks.join(' ');
    }

    function renderPositions(items) {
      const rows = (items || []).map((p) => `
        <tr>
          <td>${p.symbol || '-'}</td>
          <td>${p.direction || '-'}</td>
          <td>${fmtNum(p.entry_price)}</td>
          <td>${p.current_price === null || p.current_price === undefined ? '-' : fmtNum(p.current_price)}</td>
          <td class="${Number(p.pnl || 0) >= 0 ? 'status-ok' : 'status-bad'}">${fmtPnl(p.pnl || 0)}</td>
          <td>${p.strategy || '-'}</td>
        </tr>
      `);
      document.getElementById('positionsBody').innerHTML = rows.length ? rows.join('') : '<tr><td colspan="6">No open positions</td></tr>';
    }

    function renderLogs(logs) {
      const lines = (logs || []).map((r) => `[${r.ts}] ${r.message}`);
      document.getElementById('logs').textContent = lines.join('\\n');
    }

    async function refresh() {
      try {
        const [stateRes, posRes, logsRes] = await Promise.all([
          fetch('/api/state'),
          fetch('/api/positions'),
          fetch('/api/logs')
        ]);
        const state = await stateRes.json();
        const posData = await posRes.json();
        const logsData = await logsRes.json();
        renderState(state);
        renderPositions(posData.positions || []);
        renderLogs(logsData.logs || []);
      } catch (e) {
        document.getElementById('logs').textContent = `Dashboard fetch error: ${e}`;
      }
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class DashboardState:
    """Thread-safe in-memory dashboard event log."""

    def __init__(self, max_logs: int = 250):
        self._lock = threading.Lock()
        self._logs: Deque[Dict[str, Any]] = deque(maxlen=max_logs)

    def add_log(self, message: str) -> None:
        with self._lock:
            self._logs.append(
                {
                    "ts": time.strftime("%H:%M:%S", time.localtime()),
                    "message": str(message),
                }
            )

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._logs)[-limit:]


def create_dashboard_app(
    state_provider: Callable[[], Dict[str, Any]],
    positions_provider: Callable[[], List[Dict[str, Any]]],
    logs_provider: Callable[[], List[Dict[str, Any]]],
) -> Flask:
    """Build Flask app for the local control-room dashboard."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(_DASHBOARD_HTML)

    @app.get("/api/state")
    def api_state():
        return jsonify(state_provider())

    @app.get("/api/positions")
    def api_positions():
        return jsonify({"positions": positions_provider()})

    @app.get("/api/logs")
    def api_logs():
        return jsonify({"logs": logs_provider()})

    return app


def start_dashboard_server(
    state_provider: Callable[[], Dict[str, Any]],
    positions_provider: Callable[[], List[Dict[str, Any]]],
    logs_provider: Callable[[], List[Dict[str, Any]]],
    host: str = "127.0.0.1",
    port: int = 5018,
) -> None:
    """Run the local dashboard server."""
    app = create_dashboard_app(
        state_provider=state_provider,
        positions_provider=positions_provider,
        logs_provider=logs_provider,
    )
    logger.info(f"🌐 Dashboard available at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
