from unittest.mock import MagicMock

from dashboard.app import DashboardState, create_dashboard_app, start_dashboard_server


def test_dashboard_root_renders():
    app = create_dashboard_app(
        state_provider=lambda: {"system_running": True},
        positions_provider=lambda: [],
        logs_provider=lambda: [],
    )
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Sala di controllo V18" in response.data
    assert b"/api/state" in response.data


def test_dashboard_api_endpoints_return_expected_payload():
    state = {
        "system_running": True,
        "paper_trading": True,
        "balance": 1000.0,
        "global_win_rate": 0.6,
        "total_pnl": 15.5,
        "agent_weights": {"pattern": 1.1},
    }
    positions = [{"symbol": "BTCUSDT", "entry_price": 10.0, "current_price": 11.0, "pnl": 1.0}]
    logs = [{"ts": "12:00:00", "message": "hello"}]

    app = create_dashboard_app(
        state_provider=lambda: state,
        positions_provider=lambda: positions,
        logs_provider=lambda: logs,
    )
    client = app.test_client()

    state_response = client.get("/api/state")
    positions_response = client.get("/api/positions")
    logs_response = client.get("/api/logs")

    assert state_response.status_code == 200
    assert state_response.get_json() == state
    assert positions_response.status_code == 200
    assert positions_response.get_json() == {"positions": positions}
    assert logs_response.status_code == 200
    assert logs_response.get_json() == {"logs": logs}


def test_dashboard_state_log_buffer_keeps_latest_items():
    state = DashboardState(max_logs=2)
    state.add_log("first")
    state.add_log("second")
    state.add_log("third")

    logs = state.get_logs()
    assert len(logs) == 2
    assert logs[0]["message"] == "second"
    assert logs[1]["message"] == "third"


def test_dashboard_server_uses_v18_default_port(monkeypatch):
    run_mock = MagicMock()
    app_mock = MagicMock(run=run_mock)
    monkeypatch.setattr("dashboard.app.create_dashboard_app", lambda **_kwargs: app_mock)

    start_dashboard_server(
        state_provider=lambda: {},
        positions_provider=lambda: [],
        logs_provider=lambda: [],
    )

    run_mock.assert_called_once_with(
        host="127.0.0.1",
        port=5018,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
