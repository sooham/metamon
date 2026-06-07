# Product Requirements Document: Synchronized Showdown Replay Viewer

## 1. Overview

Currently, `inspect_replay.py` provides a terminal-based inspector for Pokémon Showdown battles — stepping through raw protocol logs, forward-filled spectator state, and RL training data turn by turn. Users navigate with `n`/`p`/`j<N>` keys.

This feature adds a synchronized browser window showing the **real Pokémon Showdown replay viewer** at `replay.pokemonshowdown.com`, controlled bidirectionally from the terminal. When the user presses `n` in the terminal, the browser advances one turn. When they click "next" in the browser, the terminal state advances too.

The browser uses the actual Showdown UI — no custom replay viewer is built. Control is achieved via the **Chrome DevTools Protocol (CDP)**, executing JavaScript and dispatching input events into the replay tab.

## 2. User Stories

| ID | Story |
|----|-------|
| **US-1** | As a metamon developer, I run `inspect_replay.py --showdown gen4uu-184050323` and a Chrome tab opens showing the real Showdown replay for that battle. |
| **US-2** | When I press `n` / `p` / `j 12` in the terminal, the browser replay navigates to the corresponding turn within ~200ms. |
| **US-3** | When I click "next turn" or use arrow keys in the browser, the terminal also advances to that turn. |
| **US-4** | I can quit the terminal (`q`) and the browser tab closes (or detaches cleanly). |
| **US-5** | If the battle does not exist on `replay.pokemonshowdown.com`, I get a clear error and the tool falls back to terminal-only mode. |
| **US-6** | The `--showdown` flag works with `--summary` and `--raw-only` modes (read-only browser, no sync needed). |

## 3. Functional Requirements

| ID | Requirement | Priority |
|----|------------|----------|
| **FR-1** | On `--showdown`, open `https://replay.pokemonshowdown.com/<battle_id>` in Chrome via CDP | P0 |
| **FR-2** | Terminal `n`/`p` maps to browser "next turn" / "previous turn" | P0 |
| **FR-3** | Terminal `j <N>` maps to browser "jump to turn N" | P0 |
| **FR-4** | Browser → terminal sync: when user navigates in browser, terminal advances to same turn (optional, P1) | P1 |
| **FR-5** | `q` in terminal closes the browser tab (or disconnects cleanly) | P0 |
| **FR-6** | Auto-detect whether Chrome is reachable on the debug port; if not, print instructions and fall back | P1 |
| **FR-7** | Auto-launch a dedicated Chrome instance with `--remote-debugging-port` if none is available (macOS only initially) | P2 |
| **FR-8** | If battle ID is not found on replay.pokemonshowdown.com (HTTP 404), inform user and fall back | P1 |

## 4. Non-Functional Requirements

| ID | Requirement |
|----|------------|
| **NFR-1** | Terminal → browser latency < 300ms for turn navigation |
| **NFR-2** | No new mandatory Python dependencies beyond `websockets` (already added) |
| **NFR-3** | Works on macOS (primary). Linux support follows. Windows is stretch. |
| **NFR-4** | Does not interfere with existing `--raw-only`, `--summary`, or parsed-state browsing (`a`) modes |
| **NFR-5** | Graceful degradation: if CDP connection fails at any point, terminal mode continues uninterrupted |

---

# System Design Document

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  inspect_replay.py                                           │
│                                                              │
│  ┌──────────────────────┐    ┌─────────────────────────────┐ │
│  │  Existing:            │    │  NEW: ShowdownBridge         │ │
│  │  - load raw replay    │    │                              │ │
│  │  - forward fill       │    │  ┌───────────────────────┐  │ │
│  │  - parsed data browse │    │  │ CDPClient              │  │ │
│  │  - terminal loop      │    │  │ (asyncio + websockets) │  │ │
│  │                        │    │  │                        │  │ │
│  │  main() loop:          │    │  │ - discover_tabs()      │  │ │
│  │    on 'n'/'p'/'j<N>':  │    │  │ - new_tab(url)         │  │ │
│  │      update terminal   │    │  │ - evaluate(js)         │  │ │
│  │      bridge.goto(N) ───┼────┼──│ - send(method,params)  │  │ │
│  │      poll_browser() ◄──┼────┼──│                        │  │ │
│  │                        │    │  └───────┬───────────────┘  │ │
│  └──────────────────────┘    │            │                  │ │
│                              │  ┌─────────▼─────────────┐  │ │
│                              │  │ ShowdownBridge         │  │ │
│                              │  │                        │  │ │
│                              │  │ - open_replay()        │  │ │
│                              │  │ - goto_turn(n)         │  │ │
│                              │  │ - next_turn()          │  │ │
│                              │  │ - prev_turn()          │  │ │
│                              │  │ - get_current_turn()   │  │ │
│                              │  │ - disconnect()         │  │ │
│                              │  └────────────────────────┘  │ │
│                              └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │ CDP WebSocket (ws://localhost:9222/devtools/page/<id>)
         │ Protocol: JSON-RPC 2.0 over WebSocket
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Google Chrome (--remote-debugging-port=9222)                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Tab: replay.pokemonshowdown.com/<battle_id>          │   │
│  │                                                       │   │
│  │  window.battle  (global battle object)                │   │
│  │  ├── .turn            → int (current turn, 0-based)   │   │
│  │  ├── .seekTurn(N)     → jump to turn N                │   │
│  │  ├── .skipTurn()      → advance one turn              │   │
│  │  ├── .setTurn(N)      → set turn without animation    │   │
│  │  ├── .started         → bool (True after Play click)  │   │
│  │  ├── .stepQueue       → Array of battle steps         │   │
│  │  └── .scene           → BattleScene renderer          │   │
│  │                                                       │   │
│  │  DOM controls:                                        │   │
│  │  <button> "Play"           — starts replay            │   │
│  │  <button> "First turn"     — jump to turn 0           │   │
│  │  <button> "Prev turn"      — previous turn            │   │
│  │  <button> "Skip turn"      — next turn                │   │
│  │  <button> "Skip to end"    — jump to last turn        │   │
│  │  <button> "Go to turn..."  — opens turn prompt        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 2. Component Design

### 2.1 `ShowdownBridge` class

Located in `metamon/tools/showdown_bridge.py`.

```
ShowdownBridge
├── __init__(battle_id, host, port)
├── REPLAY_BASE_URL = "https://replay.pokemonshowdown.com"
├── async connect()           → verify Chrome is reachable
├── async open_replay()       → open tab, wait for battle, click Play, discover max_turn
├── async goto_turn(n)        → battle.seekTurn(n), clamped to [0, max_turn]
├── async next_turn()         → goto_turn(current + 1)
├── async prev_turn()         → goto_turn(current - 1)
├── async get_current_turn()  → read battle.turn (returns -1 if inactive)
├── async disconnect()        → close tab + CDP websocket
├── max_turn (property)       → cached max turn from open_replay, or None
└── is_active (property)      → True after open_replay() succeeds
```

### 2.2 `CDPClient` class

Thin wrapper around the Chrome DevTools Protocol.  All HTTP calls use
`urllib.request` in `asyncio.to_thread` (no new dependency beyond `websockets`).

```
CDPClient
├── __init__(host, port)
├── async discover_tabs()              → HTTP GET /json → list of page tabs
├── async new_tab(url)                 → PUT /json/new, then Page.navigate + wait for load
├── async connect_tab(ws_url)          → WebSocket connect + enable Runtime/Page domains
├── async close_tab(tab_id)            → HTTP GET /json/close/<id>
├── async send(method, params)         → JSON-RPC command, matched response via id
├── async evaluate(js)                 → Runtime.evaluate wrapper
├── async dispatch_key(key, ...)       → Input.dispatchKeyEvent (keydown + keyup)
├── async click(selector)              → mouse click on a CSS selector
└── async close()                      → close WebSocket, cancel reader, reject pending
```

**Internal details**

- `_http_json(method, url)` — synchronous HTTP helper (run in thread pool).
  Returns parsed JSON, or the raw text if the response is not valid JSON
  (some CDP endpoints like `/json/close/` return plain text).
- `_read_loop()` — background asyncio task that reads every WebSocket
  message and resolves the matching `asyncio.Future` by `id`.  Events
  (messages without an `id`) are dispatched to `_event_listeners`
  keyed by CDP method name.
- `_event_listeners` — dict of `method → List[Future]` used to wait for
  specific CDP events (e.g. `Page.loadEventFired`).
- `_wait_for_page_load()` — waits for `Page.loadEventFired` via the
  event listener mechanism (called automatically by `new_tab`).

### 2.3 Modifications to `main()`

- New CLI arg: `--showdown` (flag) + `--chrome-port` (default 9222)
- Before interactive loop: instantiate `ShowdownBridge`, call `connect()` + `open_replay()`
- In the command-dispatch block (after `n`/`p`/`j<N>`/`q`): call bridge method
- On KeyboardInterrupt / exit: `await bridge.disconnect()`

## 3. Protocol Specifications

### 3.1 Chrome DevTools Protocol (CDP)

**Discovery** — HTTP REST on `http://localhost:9222`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/json` | GET | List open tabs. Each tab has `id`, `url`, `webSocketDebuggerUrl` |
| `/json/new` | PUT | Open a blank tab, return tab object. The target URL is loaded via `Page.navigate` after connecting the WebSocket. |
| `/json/close/<id>` | GET | Close tab by ID |

**Command channel** — WebSocket at `webSocketDebuggerUrl`:

Both directions use JSON-RPC 2.0 messages:
```json
{
  "id": <int>,
  "method": "<Domain.method>",
  "params": { <object> }
}
```

Response:
```json
{
  "id": <int>,
  "result": { <object> }
}
```

**Key CDP methods used:**

| Domain.method | Purpose |
|--------------|---------|
| `Runtime.evaluate` | Execute JS in the page context (primary navigation mechanism) |
| `Runtime.enable` | Enable runtime domain (needed for execution contexts) |
| `Page.navigate` | Navigate tab to a URL (used by `new_tab` to load the target URL) |
| `Page.enable` | Enable Page domain (needed for `Page.loadEventFired` events) |
| `Input.dispatchKeyEvent` | Simulate keyboard input (available, but not used for navigation) |
| `Input.dispatchMouseEvent` | Simulate mouse clicks (used by `click()` helper) |

### 3.2 Turn Navigation Implementation

**Challenge**: The Showdown replay viewer's turn state is internal to the
`BattlePanel` Preact component.  The URL hash does NOT encode the turn
number (hash is used for SPA routing: left panel vs. right panel).

**Solution — Direct API call on the global `window.battle` object.**

During DOM exploration of a live replay page we discovered that Showdown
exposes a rich global object at `window.battle` with exactly the methods
we need.  No DOM manipulation or keyboard simulation is required.

#### Reading the current turn

```javascript
// Runtime.evaluate — battle.turn is 0 for preamble, 1+ for action turns
battle.turn
```

#### Jumping to turn N

```javascript
battle.seekTurn(N)   // animated seek to turn N
battle.setTurn(N)    // instant jump (no animation)
```

Both work even before `battle.started` is `True`.

#### Advancing / rewinding one turn

```javascript
battle.seekTurn(battle.turn + 1)   // next
battle.seekTurn(battle.turn - 1)   // prev
```

`battle.skipTurn()` also exists but `seekTurn(current ± 1)` is used
for symmetry and simpler clamping.

#### Discovering the total turn count

```javascript
battle.seekTurn(9999)   // seeks to the last turn (clamped internally)
battle.turn             // now equals the maximum turn number
```

#### Starting playback

`battle.started` is `False` until the user (or a script) clicks the
"Play" button.  The bridge does this automatically during `open_replay()`:

```javascript
// Click the Play button (identified by the .fa-play icon)
document.querySelector('button .fa-play').parentElement.click()
```

**Summary**: The original three-mechanism fallback chain (DOM selector
→ Preact internals → keyboard simulation) was replaced by a single
reliable mechanism: calling `battle.seekTurn()`.  This was discovered
by probing the live replay page via CDP's `Runtime.evaluate`.

#### Why keyboard / DOM approaches were unnecessary

| Planned mechanism | Why it wasn't needed |
|-------------------|---------------------|
| Keyboard simulation (A) | `battle.seekTurn()` is more direct and works for arbitrary turns, not just ±1 |
| DOM turn-selector (B) | There is no `.turn-selector` element; turn state lives in JavaScript, not the DOM |
| Preact internals (C) | `window.battle` is a plain global — no need to walk the component tree |

### 3.3 Browser → Terminal Sync (Polling)

For US-3 (bidirectional sync), the Python script polls the browser every ~500ms:

```python
current_browser_turn = await bridge.get_current_turn()
if current_browser_turn != last_known_turn:
    # User navigated in browser; update terminal
    current_turn = current_browser_turn
    # re-render terminal display for new turn
```

This is done in a background asyncio task. The terminal input loop uses `select()` or a non-blocking read with `asyncio` integration.

## 4. Interface Definitions

### 4.1 CLI Interface

```
python tools/inspect_replay.py <battle_id> [--showdown] [--chrome-port PORT]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `battle_id` | str | required | Battle ID (e.g. `gen4uu-184050323`) |
| `--showdown` | flag | False | Enable synchronized Showdown replay viewer |
| `--chrome-port` | int | 9222 | Chrome remote debugging port |
| `--no-sync` | flag | False | Open replay in browser but don't sync navigation |

### 4.2 Python API (for potential reuse)

```python
from metamon.tools.showdown_bridge import ShowdownBridge

bridge = ShowdownBridge("gen4uu-184050323")
await bridge.connect()
await bridge.open_replay()
await bridge.goto_turn(5)
await bridge.next_turn()
turn = await bridge.get_current_turn()
await bridge.disconnect()
```

### 4.3 CDP Connection Lifecycle

```
  Python                              Chrome
    │                                    │
    │──── HTTP GET /json ───────────────►│  (discover tabs)
    │◄─── [{id, url, wsUrl}, ...] ──────│
    │                                    │
    │──── HTTP PUT /json/new?url=... ───►│  (open replay tab)
    │◄─── {id, wsUrl} ──────────────────│
    │                                    │
    │──── WS connect to wsUrl ──────────►│  (CDP session)
    │◄─── WS connected ─────────────────│
    │                                    │
    │──── Runtime.enable ───────────────►│
    │──── Page.enable ──────────────────►│
    │                                    │
    │  ... navigation commands ...       │
    │                                    │
    │──── WS close ─────────────────────►│  (on quit)
    │──── HTTP GET /json/close/<id> ────►│  (close tab)
```

### 4.4 Error Handling Matrix

| Error Condition | Behavior |
|----------------|----------|
| Chrome not reachable on port | Print: "Chrome not found on localhost:9222. Start Chrome with: `open -a 'Google Chrome' --args --remote-debugging-port=9222`" — continue in terminal-only mode |
| 404 on replay URL | Print: "Battle '<id>' not found on replay.pokemonshowdown.com" — continue terminal-only |
| CDP WebSocket disconnects mid-session | Print warning, attempt reconnect once, otherwise continue terminal-only |
| `goto_turn` fails (DOM element not found) | Print: "Could not sync turn N (replay viewer DOM not ready)" — skip, continue |
| Browser tab closed by user | Print: "Browser tab was closed. Use --showdown again to reopen." — continue terminal-only |

## 5. Data Flow

```
Terminal: user presses "n"
  │
  ▼
main() loop: current_turn += 1
  │
  ├──► Update terminal display (existing code)
  │
  └──► if bridge.is_active:
         await bridge.next_turn()
           │
           ▼
         bridge.get_current_turn()  →  read battle.turn
         bridge.goto_turn(current + 1)
           │
           ▼
         CDPClient.evaluate("battle.seekTurn(N)")
           │
           ▼
         Chrome → battle.seekTurn(N) → re-render sprites/HP bars
```

```
Browser: user clicks "Skip turn" button (or presses arrow key)
  │
  ▼
Chrome → BattlePanel → battle.turn updates
  │
  ▼
Python polling task (every 500ms, Phase 3):
  bridge.get_current_turn() → 6 (was 5)
  │
  ▼
Update terminal current_turn = 6
Re-render terminal display
```

## 6. Implementation Plan

### Phase 1: CDP Client (foundational) ✅ COMPLETE

1. ~~Create `metamon/tools/showdown_bridge.py`~~
2. Implement `CDPClient` class:
   - `discover_tabs()` — HTTP GET `/json`
   - `new_tab(url)` — HTTP PUT `/json/new` (blank tab), then `Page.navigate` + wait for `Page.loadEventFired`
   - `connect_tab(ws_url)` — WebSocket connection, enables `Runtime` + `Page` domains
   - `send(method, params)` — JSON-RPC with `asyncio.Future` request/response matching
   - `evaluate(js)` — `Runtime.evaluate` wrapper with exception surfacing
   - `dispatch_key(key, ...)` — `Input.dispatchKeyEvent` with keydown/keyup pairs
   - `click(selector)` — click a DOM element via `Input.dispatchMouseEvent`
   - `close_tab(tab_id)` — HTTP GET `/json/close/<id>`
   - Background `_read_loop` for WebSocket messages with event listener dispatch

### Phase 2: ShowdownBridge ✅ COMPLETE

1. ~~Implement `ShowdownBridge` class using `CDPClient`~~
2. `open_replay()` — navigates to replay URL, polls for `window.battle`, auto-clicks Play, discovers `max_turn` by seeking to 9999, then seeks to turn 1
3. `goto_turn(n)` — calls `battle.seekTurn(n)` with clamping to `[0, max_turn]`
4. `next_turn()` / `prev_turn()` — call `goto_turn(current ± 1)`
5. `get_current_turn()` — reads `battle.turn` (returns -1 if inactive)
6. `_wait_for_battle()` — polls for `typeof battle !== 'undefined' && battle.stepQueue.length > 0`, then auto-clicks Play
7. Error handling: `CDPConnectionError` with human-readable Chrome launch instructions, timeout on replay load with battle-ID hint

### Phase 3: Integration with `inspect_replay.py`

1. Add `--showdown` and `--chrome-port` CLI args
2. Add `async` support to `main()` (or use `asyncio.run()` wrapper)
3. Before interactive loop: await `bridge.connect()` and `bridge.open_replay()`
4. In command handlers: after processing `n`/`p`/`j<N>`, call bridge method
5. On `q`: await `bridge.disconnect()`
6. Add background polling task for browser→terminal sync

### Phase 4: Polish

1. Auto-launch Chrome with debug port on macOS
2. `--no-sync` flag for read-only browser
3. Support `--summary` and `--raw-only` modes with `--showdown`
4. Tests and documentation

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **CDP over local WebSocket, not Selenium/Playwright** | No browser driver installation needed. CDP is built into Chrome. `websockets` library is the only additional Python dependency. |
| **Real Showdown replay page, not a custom viewer** | Zero UI to build. Always up-to-date with Showdown. Sprites, animations, sound all work. |
| **`battle.seekTurn()` for all navigation** | During DOM exploration we discovered `window.battle` is a plain global object with `seekTurn(N)`, `turn`, `skipTurn()`, etc. This is far simpler than the originally planned DOM-manipulation / keyboard-simulation / Preact-internals fallback chain. No fallback chain is needed — `battle.seekTurn()` works reliably even before `battle.started` is true. |
| **Auto-click Play on load** | `battle.started` is `False` until the user clicks Play. The bridge clicks it via JS so the viewer is immediately interactive. |
| **Max-turn discovery via seek-to-9999** | `battle.seekTurn(9999)` jumps to the last turn (clamped internally). Reading `battle.turn` afterwards gives the total turn count — no need to parse the embedded log. |
| **HTTP via `urllib` in thread pool, not `aiohttp`** | Keeps the dependency footprint small. CDP HTTP endpoints are simple GET/PUT to localhost — `asyncio.to_thread` + `urllib.request` is sufficient. |
| **Polling for browser→terminal sync, not event-driven** | CDP doesn't push DOM changes. We'd need `DOM.childNodeInserted` events which are complex. 500ms polling is simple and sufficient for a dev tool. |
| **Async architecture** | CDP is inherently async (WebSocket). Terminal input can be wrapped with `asyncio` to remain responsive during sync operations. |
| **Graceful degradation, never crash** | This is a dev tool enhancement. If the browser bridge fails for any reason, the core terminal inspector must continue working. |
