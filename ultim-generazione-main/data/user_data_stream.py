"""
Binance Futures User Data Stream manager.

Handles:
- listenKey creation/keepalive/refresh via REST
- private WebSocket lifecycle with reconnects
- routing ORDER_TRADE_UPDATE and ACCOUNT_UPDATE events to a callback
"""
import json
import logging
import threading
import time
from typing import Callable, Dict, Optional

import requests
import websocket

from config.settings import API_KEY, API_SECRET

logger = logging.getLogger("UserDataStream")

_REST_BASE = "https://fapi.binance.com"
_WS_BASE = "wss://fstream.binance.com/ws"
_MIN_KEEPALIVE_INTERVAL_SEC = 30 * 60


class UserDataStreamManager:
    """Manage Binance Futures private user data stream in a resilient background loop."""

    def __init__(
        self,
        on_event: Callable[[Dict], None],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rest_base_url: str = _REST_BASE,
        ws_base_url: str = _WS_BASE,
        keepalive_interval_sec: int = 30 * 60,
        request_timeout_sec: float = 10.0,
        reconnect_delay_sec: int = 3,
        max_reconnect_delay_sec: int = 60,
    ) -> None:
        self._on_event = on_event
        self._api_key = (api_key if api_key is not None else API_KEY or "").strip()
        self._api_secret = (api_secret if api_secret is not None else API_SECRET or "").strip()
        self._rest_base_url = rest_base_url.rstrip("/")
        self._ws_base_url = ws_base_url.rstrip("/")
        self._keepalive_interval_sec = max(_MIN_KEEPALIVE_INTERVAL_SEC, int(keepalive_interval_sec))
        self._request_timeout_sec = max(1.0, float(request_timeout_sec))
        self._reconnect_delay_sec = max(1, int(reconnect_delay_sec))
        self._max_reconnect_delay_sec = max(self._reconnect_delay_sec, int(max_reconnect_delay_sec))

        self._listen_key: Optional[str] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._ws_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._ws_app: Optional[websocket.WebSocketApp] = None

    def start(self) -> bool:
        """Start stream threads. Returns False if credentials/listenKey are unavailable."""
        if not self._api_key or not self._api_secret:
            logger.warning("User Data Stream disabled: missing BINANCE_API_KEY/BINANCE_API_SECRET")
            return False

        with self._state_lock:
            if self._ws_thread and self._ws_thread.is_alive():
                return True

        listen_key = self._request_listen_key()
        if not listen_key:
            logger.warning("User Data Stream disabled: unable to obtain listenKey (fallback to polling)")
            return False

        with self._state_lock:
            self._listen_key = listen_key
            self._stop_event.clear()
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop,
                daemon=True,
                name="UserDataKeepalive",
            )
            self._ws_thread = threading.Thread(
                target=self._ws_loop,
                daemon=True,
                name="UserDataWebSocket",
            )
            self._keepalive_thread.start()
            self._ws_thread.start()

        logger.info("🔐 User Data Stream started")
        return True

    def stop(self) -> None:
        """Stop stream threads and close listenKey best-effort."""
        self._stop_event.set()
        with self._state_lock:
            ws_app = self._ws_app
            listen_key = self._listen_key
        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:
                pass

        for th in (self._ws_thread, self._keepalive_thread):
            if th and th.is_alive():
                th.join(timeout=3.0)

        if listen_key:
            self._delete_listen_key(listen_key)
        with self._state_lock:
            self._listen_key = None
            self._ws_app = None

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self._api_key}

    def _request_listen_key(self) -> Optional[str]:
        response = None
        try:
            response = requests.post(
                f"{self._rest_base_url}/fapi/v1/listenKey",
                headers=self._headers(),
                timeout=self._request_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            listen_key = str(payload.get("listenKey", "")).strip()
            return listen_key or None
        except Exception as exc:
            status = getattr(response, "status_code", "n/a")
            logger.warning(f"listenKey request failed (status={status}): {exc}")
            return None

    def _keepalive_listen_key(self, listen_key: str) -> bool:
        response = None
        try:
            response = requests.put(
                f"{self._rest_base_url}/fapi/v1/listenKey",
                headers=self._headers(),
                params={"listenKey": listen_key},
                timeout=self._request_timeout_sec,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            status = getattr(response, "status_code", "n/a")
            logger.warning(f"listenKey keepalive failed (status={status}): {exc}")
            return False

    def _delete_listen_key(self, listen_key: str) -> None:
        try:
            requests.delete(
                f"{self._rest_base_url}/fapi/v1/listenKey",
                headers=self._headers(),
                params={"listenKey": listen_key},
                timeout=self._request_timeout_sec,
            )
        except Exception:
            pass

    def _keepalive_loop(self) -> None:
        while not self._stop_event.wait(self._keepalive_interval_sec):
            with self._state_lock:
                listen_key = self._listen_key
            if not listen_key:
                continue

            if self._keepalive_listen_key(listen_key):
                continue

            # Retry by creating a new key and forcing a reconnect.
            new_key = self._request_listen_key()
            if not new_key:
                continue
            with self._state_lock:
                self._listen_key = new_key
                ws_app = self._ws_app
            if ws_app is not None:
                try:
                    ws_app.close()
                except Exception:
                    pass

    def _ws_loop(self) -> None:
        delay = self._reconnect_delay_sec
        while not self._stop_event.is_set():
            with self._state_lock:
                listen_key = self._listen_key
            if not listen_key:
                listen_key = self._request_listen_key()
                if not listen_key:
                    time.sleep(delay)
                    delay = min(delay * 2, self._max_reconnect_delay_sec)
                    continue
                with self._state_lock:
                    self._listen_key = listen_key

            ws_url = f"{self._ws_base_url}/{listen_key}"

            def _on_message(_ws, message: str) -> None:
                self._handle_message(message)

            def _on_error(_ws, err: Exception) -> None:
                logger.warning(f"User Data WS error: {err}")

            def _on_close(_ws, _code, _msg) -> None:
                logger.info("User Data WS closed")

            ws_app = websocket.WebSocketApp(
                ws_url,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )

            with self._state_lock:
                self._ws_app = ws_app

            try:
                ws_app.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    ping_payload="",
                )
            except Exception as exc:
                logger.warning(f"User Data WS run_forever error: {exc}")

            if self._stop_event.is_set():
                break
            time.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay_sec)

    def _handle_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except Exception:
            logger.debug("User Data WS received non-JSON payload")
            return

        event_type = str(payload.get("e", ""))
        if event_type in {"ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE"}:
            try:
                self._on_event(payload)
            except Exception as exc:
                logger.error(f"User Data event callback error: {exc}")
            return

        if event_type == "listenKeyExpired":
            logger.warning("User Data listenKey expired — refreshing")
            new_key = self._request_listen_key()
            if new_key:
                with self._state_lock:
                    self._listen_key = new_key
                    ws_app = self._ws_app
                if ws_app is not None:
                    try:
                        ws_app.close()
                    except Exception:
                        pass
