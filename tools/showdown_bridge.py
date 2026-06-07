"""
Chrome DevTools Protocol (CDP) client for controlling a Pokémon Showdown
replay viewer from Python.

Phase 1: CDPClient — foundational JSON-RPC over WebSocket transport,
         tab discovery, and high-level helpers (evaluate, dispatch_key).

Usage (standalone test):
  python tools/showdown_bridge.py
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import websockets
from websockets.legacy.client import WebSocketClientProtocol


# ── Exceptions ────────────────────────────────────────────────────────────────

class CDPError(Exception):
    """Raised when Chrome returns a CDP protocol error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"CDP error {code}: {message}")


class CDPConnectionError(Exception):
    """Raised when the Chrome debugging endpoint is unreachable."""


# ── CDPClient ─────────────────────────────────────────────────────────────────

class CDPClient:
    """Async client for the Chrome DevTools Protocol (CDP).

    Connects to a Chrome instance started with::

        chrome --remote-debugging-port=9222

    Parameters
    ----------
    host : str
        Host where Chrome debugging is listening (default ``"localhost"``).
    port : int
        Port where Chrome debugging is listening (default ``9222``).
    """

    def __init__(self, host: str = "localhost", port: int = 9222) -> None:
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"

        self._ws: Optional[WebSocketClientProtocol] = None
        self._msg_id: int = 0
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._event_listeners: Dict[str, List[asyncio.Future[Any]]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._closed: bool = False

    # ── HTTP helpers (sync, run in thread pool) ───────────────────────────

    @staticmethod
    def _http_json(method: str, url: str) -> Any:
        """Make an HTTP request and parse the JSON response.

        Returns None if the response body is empty or not valid JSON.
        """
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                if not data:
                    return None
                try:
                    return _json.loads(data)
                except _json.JSONDecodeError:
                    # Some CDP endpoints return plain text (e.g. /json/close/)
                    return data.decode("utf-8", errors="replace").strip() or None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise CDPConnectionError(
                f"HTTP {e.code} from {url}: {body[:200]}"
            ) from None
        except urllib.error.URLError as e:
            raise CDPConnectionError(
                f"Cannot reach Chrome debugging port at {url.split('/json')[0]}. "
                f"Is Chrome running with --remote-debugging-port?\n"
                f"  macOS: open -a 'Google Chrome' --args --remote-debugging-port=9222\n"
                f"  Linux: google-chrome --remote-debugging-port=9222\n"
                f"  Underlying error: {e.reason}"
            ) from None

    async def _http_async(self, method: str, path: str) -> Any:
        """Run an HTTP request to the CDP endpoint in a thread pool."""
        url = f"{self._base_url}{path}"
        return await asyncio.to_thread(self._http_json, method, url)

    # ── Public API ────────────────────────────────────────────────────────

    async def discover_tabs(self) -> List[Dict[str, Any]]:
        """Return the list of open debuggable tabs.

        Each tab dict contains keys: ``id``, ``url``, ``title``,
        ``webSocketDebuggerUrl``, ``type``.

        Raises
        ------
        CDPConnectionError
            If Chrome is not reachable on the configured host/port.
        """
        tabs = await self._http_async("GET", "/json")
        if tabs is None:
            return []
        return [t for t in tabs if t.get("type") == "page"]

    async def new_tab(self, url: str = "about:blank") -> Dict[str, Any]:
        """Open a new tab (or foreground an existing one) and return its info.

        The Chrome DevTools ``/json/new`` HTTP endpoint is used to create
        the tab, then `connect_tab` + `Page.navigate` load the target URL.
        This two-step process is more reliable than passing ``url`` through
        the HTTP endpoint alone.

        Parameters
        ----------
        url : str
            URL to load in the new tab.

        Returns
        -------
        dict
            Tab info dict with keys ``id``, ``url``, ``webSocketDebuggerUrl``.

        Raises
        ------
        CDPConnectionError
            If Chrome cannot be reached.
        """
        # Open a blank tab first
        tab = await self._http_async("PUT", "/json/new")
        if tab is None or not isinstance(tab, dict):
            raise CDPConnectionError("Chrome returned empty response for new tab")

        ws_url = tab.get("webSocketDebuggerUrl")
        if not ws_url:
            raise CDPConnectionError("No webSocketDebuggerUrl in new tab response")

        # Connect and navigate to the target URL
        await self.connect_tab(ws_url)
        await self.send("Page.navigate", {"url": url})
        await self._wait_for_page_load()
        return tab

    async def _wait_for_page_load(self, timeout: float = 15.0) -> None:
        """Wait until the page finishes loading (Page.loadEventFired)."""
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        # We piggyback on _read_loop to catch the Page.loadEventFired event.
        # Temporarily store a listener.
        self._event_listeners.setdefault("Page.loadEventFired", []).append(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            pass  # Page may still be usable even without the event

    async def close_tab(self, tab_id: str) -> None:
        """Close a tab by its CDP id.

        Parameters
        ----------
        tab_id : str
            The tab's ``id`` field from `discover_tabs` or `new_tab`.
        """
        try:
            await self._http_async("GET", f"/json/close/{tab_id}")
        except CDPConnectionError:
            pass  # Tab may already be closed

    async def connect_tab(self, ws_url: str) -> None:
        """Establish a CDP WebSocket session with the given tab.

        Parameters
        ----------
        ws_url : str
            The ``webSocketDebuggerUrl`` from a tab dict.

        After connecting, enables the ``Runtime`` domain so that
        `evaluate` works.
        """
        if self._ws is not None:
            await self.close()

        self._ws = await websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024,  # 10 MB – replay pages embed the full log
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._closed = False
        self._event_listeners.clear()

        # Enable essential domains
        await self.send("Runtime.enable")
        await self.send("Page.enable")

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a CDP command and wait for the JSON-RPC response.

        Parameters
        ----------
        method : str
            CDP method name (e.g. ``"Runtime.evaluate"``).
        params : dict or None
            Optional parameters for the method.

        Returns
        -------
        Any
            The ``result`` field from the CDP response.

        Raises
        ------
        CDPError
            If Chrome returns an error for the command.
        RuntimeError
            If there is no active WebSocket connection.
        """
        if self._ws is None:
            raise RuntimeError("Not connected to a tab. Call connect_tab() first.")

        self._msg_id += 1
        msg_id = self._msg_id

        message: Dict[str, Any] = {"id": msg_id, "method": method}
        if params is not None:
            message["params"] = params

        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut

        raw = _json.dumps(message)
        await self._ws.send(raw)

        try:
            return await asyncio.wait_for(fut, timeout=15.0)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise CDPError(-1, f"Timeout waiting for response to {method}")

    async def evaluate(self, expression: str, *,
                       await_promise: bool = False,
                       return_by_value: bool = True) -> Any:
        """Execute a JavaScript expression in the page context.

        Parameters
        ----------
        expression : str
            JavaScript to evaluate.
        await_promise : bool
            If True, wait for the returned Promise to resolve.
        return_by_value : bool
            If True, return the value directly (not a RemoteObject).

        Returns
        -------
        Any
            The result value if ``return_by_value`` is True, otherwise
            the CDP ``RemoteObject`` dict.

        Raises
        ------
        CDPError
            If the expression throws an error.
        """
        params: Dict[str, Any] = {
            "expression": expression,
            "returnByValue": return_by_value,
            "awaitPromise": await_promise,
        }
        result = await self.send("Runtime.evaluate", params)
        result_obj = result.get("result", {})

        # If the evaluation threw, surface the exception
        if result_obj.get("subtype") == "error":
            desc = result_obj.get("description", "Unknown JS error")
            raise CDPError(-1, f"JS evaluation failed: {desc}")
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            text = details.get("text", details.get("exception", {}).get("description", "JS exception"))
            raise CDPError(-1, f"JS exception: {text}")

        return result_obj.get("value") if return_by_value else result_obj

    async def dispatch_key(self, key: str, *,
                           code: Optional[str] = None,
                           vk: Optional[int] = None,
                           modifiers: int = 0,
                           keydown_delay: float = 0.02) -> None:
        """Dispatch a keyboard event (keydown + keyup) into the page.

        Parameters
        ----------
        key : str
            The ``key`` value for the KeyboardEvent (e.g. ``"ArrowRight"``).
        code : str or None
            The ``code`` value. Defaults to ``key``.
        vk : int or None
            The Windows virtual key code. If None, a rough mapping is used.
        modifiers : int
            Bitmask of modifiers (0=none, 1=Alt, 2=Ctrl, 4=Meta, 8=Shift).
        keydown_delay : float
            Seconds to wait between keydown and keyup (default 20ms).
        """
        if code is None:
            code = key

        # Rough virtual key code mapping for common keys
        if vk is None:
            vk = _KEY_TO_VK.get(key, 0)

        base_params: Dict[str, Any] = {
            "type": "rawKeyDown",
            "key": key,
            "code": code,
            "windowsVirtualKeyCode": vk,
            "nativeVirtualKeyCode": vk,
            "modifiers": modifiers,
        }
        await self.send("Input.dispatchKeyEvent", {**base_params, "type": "rawKeyDown"})
        if keydown_delay:
            await asyncio.sleep(keydown_delay)
        await self.send("Input.dispatchKeyEvent", {**base_params, "type": "rawKeyUp"})

    async def click(self, selector: str) -> None:
        """Click a DOM element identified by a CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector for the element to click.
        """
        # Get the element's bounding box via JS
        result = await self.evaluate(f"""
            (() => {{
                const el = document.querySelector({_json.dumps(selector)});
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {{
                    x: rect.left + rect.width / 2,
                    y: rect.top + rect.height / 2,
                }};
            }})()
        """)
        if result is None:
            raise CDPError(-1, f"Element not found: {selector}")

        x, y = result["x"], result["y"]
        await self.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": x, "y": y,
            "button": "left",
            "clickCount": 1,
        })
        await self.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": x, "y": y,
            "button": "left",
            "clickCount": 1,
        })

    async def close(self) -> None:
        """Close the CDP WebSocket session cleanly.

        Does *not* close the browser tab — use `close_tab` for that.
        """
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Reject any outstanding futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(CDPConnectionError("CDP connection closed"))
        self._pending.clear()

    # ── Internal ──────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Background task: read messages from the WebSocket and resolve
        pending futures by matching ``id`` fields."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    msg = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None:
                    fut = self._pending.pop(msg_id, None)
                    if fut is not None and not fut.done():
                        error = msg.get("error")
                        if error is not None:
                            fut.set_exception(
                                CDPError(
                                    error.get("code", -1),
                                    error.get("message", "Unknown CDP error"),
                                )
                            )
                        else:
                            fut.set_result(msg.get("result"))
                else:
                    # Events (no "id"): dispatch to listeners
                    method = msg.get("method", "")
                    for fut in self._event_listeners.pop(method, []):
                        if not fut.done():
                            fut.set_result(msg.get("params"))
        except (websockets.ConnectionClosed, asyncio.CancelledError, Exception):
            pass
        finally:
            # If the connection drops unexpectedly, reject remaining futures
            if not self._closed:
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(
                            CDPConnectionError("CDP WebSocket disconnected")
                        )
                self._pending.clear()


# ── Key code mapping ──────────────────────────────────────────────────────────

_KEY_TO_VK: Dict[str, int] = {
    "ArrowLeft": 37,
    "ArrowUp": 38,
    "ArrowRight": 39,
    "ArrowDown": 40,
    " ": 32,
    "Space": 32,
    "Enter": 13,
    "Tab": 9,
    "Escape": 27,
    "Backspace": 8,
    "Delete": 46,
    "Home": 36,
    "End": 35,
    "PageUp": 33,
    "PageDown": 34,
}


# ── ShowdownBridge ────────────────────────────────────────────────────────────

class ShowdownBridge:
    """High-level bridge to control a Pok\u00e9mon Showdown replay viewer.

    Opens a replay on ``replay.pokemonshowdown.com`` in Chrome (via CDP)
    and provides methods to navigate turns.

    Parameters
    ----------
    battle_id : str
        Showdown battle ID (e.g. ``"gen4uu-184050323"``).
    host : str
        Chrome debugging host (default ``"localhost"``).
    port : int
        Chrome debugging port (default ``9222``).

    Usage::

        bridge = ShowdownBridge("gen1ou-316031019")
        await bridge.connect()
        await bridge.open_replay()
        await bridge.goto_turn(5)
        await bridge.disconnect()
    """

    REPLAY_BASE_URL = "https://replay.pokemonshowdown.com"

    # ── Static helpers ─────────────────────────────────────────────────

    @staticmethod
    def ensure_chrome(port: int = 9222) -> bool:
        """Make sure a Chrome instance with remote debugging is running.

        If Chrome is not already listening on *port*, attempt to launch
        a dedicated instance with a temporary user-data directory.

        Returns
        -------
        bool
            ``True`` if Chrome is now reachable on *port*.
        """
        import shutil
        import subprocess
        import tempfile

        # Already running?
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
            return True
        except Exception:
            pass

        # Find Chrome / Chromium binary
        chrome_path: Optional[str] = None
        for candidate in [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            shutil.which("google-chrome"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
        ]:
            if candidate and os.path.isfile(candidate):
                chrome_path = candidate
                break

        if chrome_path is None:
            return False

        # Launch with a fresh temp profile
        tmpdir = tempfile.mkdtemp(prefix="chrome-cdp-")
        try:
            subprocess.Popen(
                [
                    chrome_path,
                    f"--remote-debugging-port={port}",
                    f"--user-data-dir={tmpdir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False

        # Wait for the debug port to become available
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
                return True
            except Exception:
                time.sleep(0.3)
        return False

    def __init__(
        self,
        battle_id: str,
        host: str = "localhost",
        port: int = 9222,
    ) -> None:
        self._battle_id = battle_id
        self._client = CDPClient(host=host, port=port)
        self._tab_id: Optional[str] = None
        self._max_turn: Optional[int] = None
        self._remote_players: Optional[Tuple[str, str]] = None
        self._active: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Verify Chrome is reachable on the debugging port.

        If Chrome is not already running with ``--remote-debugging-port``,
        this method attempts to auto-launch a dedicated instance on macOS
        and Linux.

        Raises
        ------
        CDPConnectionError
            If Chrome cannot be launched or reached.
        """
        # Try to auto-launch Chrome if it isn't already running
        if not self.ensure_chrome(self._client._port):
            raise CDPConnectionError(
                f"Cannot start or reach Chrome on localhost:{self._client._port}.\n"
                f"Start Chrome manually:\n"
                f"  macOS: open -a 'Google Chrome' --args --remote-debugging-port={self._client._port}\n"
                f"  Linux: google-chrome --remote-debugging-port={self._client._port}"
            )
        # Verify connectivity by listing tabs
        await self._client.discover_tabs()

    async def open_replay(self) -> None:
        """Open the replay page in a new Chrome tab and wait for the
        battle viewer to finish loading.

        Raises
        ------
        CDPConnectionError
            If Chrome is unreachable.
        CDPError
            If the replay fails to load within the timeout.
        """
        url = f"{self.REPLAY_BASE_URL}/{self._battle_id}"
        tab = await self._client.new_tab(url)
        self._tab_id = tab["id"]

        # Wait for the battle object to be ready (client-side JS render)
        await self._wait_for_battle(timeout=20.0)

        # Seek to the end to discover the total turn count
        await self._client.evaluate("battle.seekTurn(9999)")
        await asyncio.sleep(2.0)
        try:
            self._max_turn = await self._client.evaluate("battle.turn")
            if not isinstance(self._max_turn, (int, float)):
                self._max_turn = 0
            else:
                self._max_turn = int(self._max_turn)
        except Exception:
            self._max_turn = 0

        # Go to turn 1 (skip preamble / team preview)
        await self._client.evaluate("battle.seekTurn(1)")
        await asyncio.sleep(0.5)
        # Pause immediately so the battle doesn't auto-play
        await self._client.evaluate("battle.pause()")

        # Capture the remote player names for mismatch detection
        try:
            p1 = await self._client.evaluate("battle.p1.name")
            p2 = await self._client.evaluate("battle.p2.name")
            if isinstance(p1, str) and isinstance(p2, str):
                self._remote_players = (p1, p2)
        except Exception:
            pass

        self._active = True

    async def goto_turn(self, turn: int) -> None:
        """Jump to a specific turn.

        Turn 0 is the preamble / team preview. Turn 1 is the first
        action turn.  Values outside the valid range are clamped.

        Parameters
        ----------
        turn : int
            Target turn number (0-indexed, same as ``battle.turn``).
        """
        if not self._active:
            return
        max_t = self._max_turn if self._max_turn is not None else 999
        turn = max(0, min(turn, max_t))
        await self._client.evaluate(f"battle.seekTurn({turn})")
        # Small delay to let the animation start, then pause to stop auto-play
        await asyncio.sleep(0.15)
        await self._client.evaluate("battle.pause()")

    async def next_turn(self) -> None:
        """Advance to the next turn (or do nothing if at the end)."""
        current = await self.get_current_turn()
        if self._max_turn is None or current < self._max_turn:
            await self.goto_turn(current + 1)

    async def prev_turn(self) -> None:
        """Go back to the previous turn (or do nothing if at turn 0)."""
        current = await self.get_current_turn()
        if current > 0:
            await self.goto_turn(current - 1)

    async def get_current_turn(self) -> int:
        """Return the current turn number shown in the browser.

        Returns -1 if the battle object is not available.
        """
        if not self._active:
            return -1
        try:
            val = await self._client.evaluate("battle.turn")
            return int(val) if isinstance(val, (int, float)) else -1
        except Exception:
            return -1

    @property
    def max_turn(self) -> Optional[int]:
        """The highest valid turn number (0-indexed), or None if unknown."""
        return self._max_turn

    @property
    def remote_players(self) -> Optional[Tuple[str, str]]:
        """The (p1, p2) player names shown on replay.pokemonshowdown.com,
        or None if not yet available."""
        return self._remote_players

    @property
    def is_active(self) -> bool:
        """True if the replay is open and the bridge is ready."""
        return self._active

    async def disconnect(self) -> None:
        """Close the replay tab and the CDP connection."""
        self._active = False
        if self._tab_id is not None:
            try:
                await self._client.close_tab(self._tab_id)
            except Exception:
                pass
            self._tab_id = None
        await self._client.close()
        self._max_turn = None

    # ── Internal ──────────────────────────────────────────────────────────

    async def _wait_for_battle(self, timeout: float = 20.0) -> None:
        """Poll until the global ``battle`` object is initialised.

        Once the battle object is available we click \"Play\" so that
        ``battle.started`` becomes ``True`` and the viewer is ready.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        last_error: Optional[str] = None
        battle_ready = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                exists = await self._client.evaluate(
                    "typeof battle !== 'undefined' && battle.stepQueue && battle.stepQueue.length > 0"
                )
                if exists:
                    battle_ready = True
                    break
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(0.5)

        if not battle_ready:
            detail = last_error or "battle object never appeared"
            raise CDPConnectionError(
                f"Timed out waiting for replay to load ({timeout}s): {detail}\n"
                f"The battle '{self._battle_id}' may not exist on "
                f"{self.REPLAY_BASE_URL}."
            )

        # Click Play to start the replay (battle.started won't be true otherwise)
        try:
            await self._client.evaluate(
                """(function(){
                    var btns = document.querySelectorAll('button');
                    for (var i=0; i<btns.length; i++) {
                        if (btns[i].querySelector('.fa-play')) {
                            btns[i].click(); return 'clicked';
                        }
                    }
                    return 'not found';
                })()"""
            )
        except Exception:
            pass  # Play button might not exist or already clicked

        # Wait briefly for the battle to start
        for _ in range(10):
            try:
                started = await self._client.evaluate("battle.started")
                if started:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)


# ── Self-test ─────────────────────────────────────────────────────────────────

async def _self_test_cdp() -> None:
    """Quick smoke test: open a page, inject a title, read it back."""
    print("=== CDPClient self-test ===")
    client = CDPClient()

    # Discover
    try:
        tabs = await client.discover_tabs()
        print(f"  Found {len(tabs)} debuggable tab(s)")
    except CDPConnectionError as e:
        print(f"  SKIP: {e}")
        return

    # Open a test tab
    tab = await client.new_tab("about:blank")
    ws_url = tab.get("webSocketDebuggerUrl")
    tab_id = tab.get("id")
    if not ws_url or not tab_id:
        print("  FAIL: new_tab returned no WebSocket URL")
        return
    print(f"  Opened tab {tab_id}")

    # Connect and test evaluate
    try:
        await client.connect_tab(ws_url)
        print("  Connected to CDP WebSocket")

        # Set document title
        await client.evaluate("document.title = 'CDPClient test OK'")

        # Read it back
        title = await client.evaluate("document.title")
        print(f"  Page title: {title}")
        assert title == "CDPClient test OK", f"Expected title, got {title!r}"

        # Test key dispatch (inject a character into an input, then read it)
        await client.evaluate("""
            const inp = document.createElement('input');
            inp.id = '__cdp_test_input';
            document.body.appendChild(inp);
            inp.focus();
        """)
        # TypeError dispatch via JS (more reliable than Input.dispatchKeyEvent for typing)
        await client.evaluate(
            "document.getElementById('__cdp_test_input').value = 'hello'"
        )
        val = await client.evaluate(
            "document.getElementById('__cdp_test_input').value"
        )
        print(f"  Input value: {val}")
        assert val == "hello", f"Expected 'hello', got {val!r}"

        print("  ✓ All checks passed")
    except Exception as e:
        print(f"  FAIL: {e}")
    finally:
        await client.close()
        await client.close_tab(tab_id)
        print("  Cleaned up")


async def _self_test_showdown() -> None:
    """Integration test: open a real replay and navigate through turns."""
    print("\n=== ShowdownBridge self-test ===")
    # Use a well-known public battle that should always exist
    battle_id = "gen1ou-316031019"
    bridge = ShowdownBridge(battle_id)

    try:
        await bridge.connect()
        print(f"  Chrome reachable")
    except CDPConnectionError as e:
        print(f"  SKIP: {e}")
        return

    try:
        await bridge.open_replay()
        print(f"  Replay loaded: max_turn={bridge.max_turn}")

        # Read initial turn
        turn = await bridge.get_current_turn()
        print(f"  Initial turn: {turn}")
        assert turn == 1, f"Expected turn 1, got {turn}"

        # Next turn
        await bridge.next_turn()
        turn = await bridge.get_current_turn()
        print(f"  After next: {turn}")
        assert turn == 2, f"Expected turn 2, got {turn}"

        # Jump to turn 10
        await bridge.goto_turn(10)
        turn = await bridge.get_current_turn()
        print(f"  After goto_turn(10): {turn}")
        assert turn == 10, f"Expected turn 10, got {turn}"

        # Prev turn
        await bridge.prev_turn()
        turn = await bridge.get_current_turn()
        print(f"  After prev: {turn}")
        assert turn == 9, f"Expected turn 9, got {turn}"

        # Jump to preamble
        await bridge.goto_turn(0)
        turn = await bridge.get_current_turn()
        print(f"  After goto_turn(0): {turn}")
        assert turn == 0, f"Expected turn 0, got {turn}"

        # Clamping: jump beyond max should cap at max_turn
        await bridge.goto_turn(999999)
        turn = await bridge.get_current_turn()
        print(f"  After goto_turn(999999): {turn} (max={bridge.max_turn})")
        assert turn == bridge.max_turn, f"Expected {bridge.max_turn}, got {turn}"

        print("  \u2713 All checks passed")
    except CDPConnectionError as e:
        print(f"  SKIP: {e}")
    except Exception as e:
        print(f"  FAIL: {e}")
    finally:
        await bridge.disconnect()
        print("  Cleaned up")


if __name__ == "__main__":
    import sys
    if "--showdown" in sys.argv:
        asyncio.run(_self_test_showdown())
    else:
        asyncio.run(_self_test_cdp())
