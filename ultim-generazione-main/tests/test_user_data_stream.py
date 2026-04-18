import json

from data.user_data_stream import UserDataStreamManager


def test_user_data_stream_start_returns_false_without_credentials():
    manager = UserDataStreamManager(
        on_event=lambda _: None,
        api_key="",
        api_secret="",
    )
    assert manager.start() is False


def test_user_data_stream_routes_order_and_account_events():
    events = []
    manager = UserDataStreamManager(
        on_event=lambda event: events.append(event),
        api_key="k",
        api_secret="s",
    )
    manager._handle_message(json.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"s": "BTCUSDT"}}))
    manager._handle_message(json.dumps({"e": "ACCOUNT_UPDATE", "a": {"P": []}}))
    manager._handle_message(json.dumps({"e": "MARGIN_CALL"}))

    assert [e["e"] for e in events] == ["ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE"]


def test_user_data_stream_listen_key_expired_refreshes_and_closes_socket(monkeypatch):
    manager = UserDataStreamManager(
        on_event=lambda _: None,
        api_key="k",
        api_secret="s",
    )

    class DummyWS:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    dummy_ws = DummyWS()
    manager._ws_app = dummy_ws
    monkeypatch.setattr(manager, "_request_listen_key", lambda: "new_listen_key")

    manager._handle_message(json.dumps({"e": "listenKeyExpired"}))

    assert manager._listen_key == "new_listen_key"
    assert dummy_ws.closed is True
