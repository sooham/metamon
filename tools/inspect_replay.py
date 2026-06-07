#!/usr/bin/env python3
"""
Inspect a Pokémon Showdown replay at all three stages of the metamon pipeline:
  1. RAW       – Original Showdown protocol log lines
  2. FORWARD   – Spectator POV (all visible info reconstructed via forward_fill)
  3. PARSED    – RL training data from both player perspectives (.json.lz4)

Usage:
  uv run python tools/inspect_replay.py <battle_id>
  uv run python tools/inspect_replay.py gen4uu-184050323

The script finds matching raw and parsed replay files in METAMON_CACHE_DIR
and displays them side by side, stepping turn by turn.
"""

import os
import sys
import orjson
import re
import asyncio
import argparse
import select
import termios
import tty
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import lz4.frame

from metamon.config import METAMON_CACHE_DIR

# ── Terminal raw-mode helpers (arrow key support) ───────────────────────────

# Cache the original terminal settings so we can restore on exit.
_ORIG_TERMIOS: Optional[list] = None


def _start_raw_mode() -> None:
    """Put stdin in cbreak mode: read byte-by-byte, no echo, no buffering."""
    global _ORIG_TERMIOS
    if _ORIG_TERMIOS is not None:
        return  # already in raw mode
    try:
        fd = sys.stdin.fileno()
        _ORIG_TERMIOS = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, AttributeError):
        pass  # not a real terminal (e.g. piped input)


def _stop_raw_mode() -> None:
    """Restore original terminal settings."""
    global _ORIG_TERMIOS
    if _ORIG_TERMIOS is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _ORIG_TERMIOS)
    except (termios.error, AttributeError):
        pass
    _ORIG_TERMIOS = None


def _poll_keypress() -> Optional[str]:
    """Non-blocking check for a keypress on stdin.

    Returns one of the following (or ``None`` if no input is available):

    * ``"RIGHT"``, ``"LEFT"``, ``"UP"``, ``"DOWN"`` — arrow keys
    * ``"ENTER"`` — Return / Enter
    * ``"ESC"`` — Escape
    * A single character string for printable keys

    Requires that `_start_raw_mode` has been called first.
    """
    try:
        fd = sys.stdin.fileno()
        r, _w, _x = select.select([fd], [], [], 0)
        if not r:
            return None
        ch = os.read(fd, 6)  # read up to 6 bytes (longest escape sequence)
        if not ch:
            return None

        # Arrow keys and other escape sequences start with 0x1b (ESC)
        if ch[0] == 0x1b:
            if len(ch) == 1:
                return "ESC"
            if ch[1] == 0x5b:  # '['
                if len(ch) >= 3:
                    seq = ch[2]
                    if seq == 0x41:
                        return "UP"
                    if seq == 0x42:
                        return "DOWN"
                    if seq == 0x43:
                        return "RIGHT"
                    if seq == 0x44:
                        return "LEFT"
            return None  # unknown escape, ignore

        if ch == b'\r' or ch == b'\n':
            return "ENTER"
        if ch == b'\x7f' or ch == b'\x08':
            return "BACKSPACE"
        if ch == b'\t':
            return "TAB"

        return ch.decode("utf-8", errors="replace") if len(ch) == 1 else None
    except (OSError, TypeError):
        return None


def _read_line_in_raw_mode(prompt: str = "") -> str:
    """Read a full line of input while in raw mode.

    Echoes typed characters and handles backspace.  Arrow keys are
    *not* handled here — call `_poll_keypress` first if you need them.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf: List[str] = []
    try:
        fd = sys.stdin.fileno()
        while True:
            r, _w, _x = select.select([fd], [], [])
            if not r:
                continue
            ch = os.read(fd, 16)
            if not ch:
                break
            b = ch[0]
            if b in (0x0a, 0x0d):  # Enter
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buf)
            if b in (0x7f, 0x08):  # Backspace
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif 0x20 <= b < 0x7f:  # printable
                char = chr(b)
                buf.append(char)
                sys.stdout.write(char)
                sys.stdout.flush()
    except (OSError, EOFError):
        return "".join(buf)
    return "".join(buf)

_CACHE_DIR = os.path.expanduser(METAMON_CACHE_DIR) if METAMON_CACHE_DIR else None

# ── RAW side ──────────────────────────────────────────────────────────────────
# These are lightweight and do not pull in the heavy ML stack unless needed.

def _find_raw_replay(gameid: str) -> Optional[str]:
    """Search for a raw replay JSON by gameid under METAMON_CACHE_DIR/raw-replays."""
    if _CACHE_DIR is None:
        return None
    raw_root = os.path.join(_CACHE_DIR, "raw-replays")
    for root, _dirs, files in os.walk(raw_root):
        for f in files:
            if f == f"{gameid}.json":
                return os.path.join(root, f)
    return None


def _parse_raw_turns(log: str) -> List[Tuple[int, List[str]]]:
    """Split the Showdown protocol log into (turn_number, [lines]) pairs.

    Lines before the first ``|turn|`` are grouped as turn 0 (preamble).
    """
    turns: List[Tuple[int, List[str]]] = []
    current_turn = 0
    current_lines: List[str] = []
    for line in log.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip bare "|" separator lines (no protocol content)
        if line == "|":
            continue
        m = re.match(r"^\|turn\|(\d+)", line)
        if m:
            if current_lines:
                turns.append((current_turn, current_lines))
            current_turn = int(m.group(1))
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        turns.append((current_turn, current_lines))
    return turns


def _raw_line_type(line: str) -> str:
    """Extract the protocol message type from a line (e.g. ``|move|`` → ``move``)."""
    # Bare pipe separator — not a protocol message
    if line == "|":
        return ""
    m = re.match(r"^\|([-\w]+)", line)
    return m.group(1) if m else "?"


# ── PARSED side ───────────────────────────────────────────────────────────────

def _find_parsed_replays(gameid: str) -> Dict[str, str]:
    """Search for parsed .json.lz4 files whose filename starts with *gameid*.

    Returns a dict mapping ``"WIN" | "LOSS"`` → absolute path.
    """
    if _CACHE_DIR is None:
        return {}
    parsed_root = os.path.join(_CACHE_DIR, "parsed-replays")
    results: Dict[str, str] = {}
    for root, _dirs, files in os.walk(parsed_root):
        for f in files:
            if f.startswith(f"{gameid}_") and f.endswith(".json.lz4"):
                if f.endswith("_WIN.json.lz4"):
                    results["WIN"] = os.path.join(root, f)
                elif f.endswith("_LOSS.json.lz4"):
                    results["LOSS"] = os.path.join(root, f)
    return results


def _load_parsed(path: str) -> dict:
    """Load a .json.lz4 parsed replay into a plain dict."""
    with lz4.frame.open(path, "rb") as fh:
        return orjson.loads(fh.read())


def _pokemon_summary(p: dict) -> str:
    """One-line summary of a UniversalPokemon dict."""
    moves = ", ".join(m["name"] for m in p.get("moves", [])[:4])
    if not moves:
        moves = "(no moves revealed)"
    hp = p.get("hp_pct", 1.0)
    return f"{p['name']} ({hp*100:.0f}%) [{moves}]"


def _move_summary(m: dict) -> str:
    """One-line summary of a UniversalMove dict."""
    return f"{m['name']} (bp={m['base_power']}, acc={m['accuracy']})"


def _action_to_label(state: dict, action_idx: int) -> str:
    """Convert an action index to a human-readable label."""
    if action_idx == -1:
        return "missing"
    if action_idx <= 3:
        moves = state["player_active_pokemon"]["moves"]
        if action_idx < len(moves):
            return f"move: {moves[action_idx]['name']}"
        return f"move idx {action_idx} (oob)"
    if 4 <= action_idx <= 8:
        switches = state["available_switches"]
        switch_idx = action_idx - 4
        if switch_idx < len(switches):
            return f"switch → {switches[switch_idx]['name']}"
        return f"switch idx {switch_idx} (oob)"
    if action_idx >= 9:
        moves = state["player_active_pokemon"]["moves"]
        tera_idx = action_idx - 9
        if tera_idx < len(moves):
            return f"tera + {moves[tera_idx]['name']}"
        return f"tera move idx {tera_idx} (oob)"
    return f"unknown action {action_idx}"


# ── FORWARD pass ──────────────────────────────────────────────────────────────

def _run_forward(raw_data: dict, verbose: bool = False):
    """Run the forward fill on a raw replay dict (must be called inside metamon venv)."""
    from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
    from metamon.backend.replay_parser.parse_replays import ReplayParser

    log = raw_data["log"]
    gameid = raw_data.get("id", "unknown")
    formatid = raw_data.get("formatid", "unknown")
    uploadtime = raw_data.get("uploadtime", 0)
    time_played = datetime.fromtimestamp(int(uploadtime))

    replay = ParsedReplay(
        gameid=gameid,
        format=formatid,
        time_played=time_played,
    )
    log_lines = ReplayParser.clean_log(raw_data)
    replay = forward_fill(replay, log_lines, verbose=verbose)
    return replay


def _forward_turn_summary(turn, gen: int) -> List[str]:
    """Produce a list of human-readable lines describing what happened in one Turn."""
    lines = []
    # Active Pokémon and their HP
    for is_p1, tag in [(True, "P1"), (False, "P2")]:
        active_list = turn.active_pokemon_1 if is_p1 else turn.active_pokemon_2
        move_list = turn.moves_1 if is_p1 else turn.moves_2
        for slot_idx, active in enumerate(active_list):
            if active is None:
                continue
            hp_str = f"{active.current_hp}/{active.max_hp}" if active.max_hp else "?"
            action = move_list[slot_idx] if slot_idx < len(move_list) else None
            if action is not None:
                if action.is_switch:
                    out_name = action.user.name if action.user else "?"
                    in_name = action.target.name if action.target else "?"
                    lines.append(
                        f"  {tag}: {out_name} → {in_name} (now {active.name} {hp_str})"
                    )
                elif action.is_noop:
                    lines.append(f"  {tag}: {active.name} ({hp_str}) recharges")
                elif action.is_revival:
                    lines.append(f"  {tag}: revives → {action.target.name if action.target else '?'}")
                else:
                    tgt = action.target.name if action.target else "?"
                    lines.append(
                        f"  {tag}: {active.name} ({hp_str}) uses {action.name} → {tgt}"
                    )
            else:
                lines.append(f"  {tag}: {active.name} ({hp_str}) (no action)")
    if not lines:
        lines.append("  (no actions this turn)")
    return lines


# ── Display ───────────────────────────────────────────────────────────────────

def _print_header(text: str, width: int = 60):
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def _print_turn_separator(turn_num: int):
    print(f"\n{'─' * 60}")
    print(f"  TURN {turn_num}")
    print(f"{'─' * 60}")


def _count_line_types(lines: List[str]) -> Dict[str, int]:
    """Count occurrences of each protocol line type in a list of raw lines."""
    counts: Dict[str, int] = {}
    for line in lines:
        t = _raw_line_type(line)
        counts[t] = counts.get(t, 0) + 1
    return counts


# ── Full raw-log display ("R" key) ───────────────────────────────────────────

# ANSI escape codes for colourised output.
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold_red": "\033[1;31m",
    "bold_yellow": "\033[1;33m",
    "bold_cyan": "\033[1;36m",
}

# Mapping from protocol line type → ANSI colour
_LINE_COLOURS: Dict[str, str] = {
    "turn": "bold_yellow",
    "move": "green",
    "switch": "green",
    "faint": "bold_red",
    "drag": "bold_red",
    "-damage": "red",
    "-heal": "green",
    "-status": "blue",
    "-boost": "blue",
    "-unboost": "blue",
    "-curestatus": "blue",
    "-transform": "blue",
    "-sidestart": "cyan",
    "-sideend": "cyan",
    "-fieldstart": "cyan",
    "-fieldend": "cyan",
    "-weather": "cyan",
    "-ability": "magenta",
    "-item": "magenta",
    "-enditem": "magenta",
    "player": "bold_cyan",
    "teamsize": "dim",
    "gen": "dim",
    "tier": "dim",
    "chat": "dim",
    "c": "dim",
    "c:": "dim",
    "raw": "dim",
    "inactive": "dim",
    "callback": "dim",
    "debug": "dim",
}


def _format_full_log(log_text: str) -> str:
    """Return a colourised, turn-grouped rendering of the full protocol log."""
    turns = _parse_raw_turns(log_text)
    out: List[str] = []
    a = _ANSI

    # Legend
    out.append(f"{a['bold']}Protocol log ({len(log_text.splitlines())} lines, {len(turns)} turns){a['reset']}")
    out.append(
        f"  {a['bold_yellow']}turn{a['reset']}  "
        f"{a['green']}move/switch{a['reset']}  "
        f"{a['red']}-damage{a['reset']}  "
        f"{a['blue']}-status/boost{a['reset']}  "
        f"{a['cyan']}field/weather{a['reset']}  "
        f"{a['dim']}other{a['reset']}"
    )
    out.append("─" * 72)

    for turn_num, lines in turns:
        if turn_num == 0:
            header = f"{a['bold']}── Preamble ──{a['reset']}"
        else:
            header = f"{a['bold_yellow']}── Turn {turn_num} ──{a['reset']}"
        out.append(f"\n{header}")

        for line in lines:
            ltype = _raw_line_type(line)
            colour = _LINE_COLOURS.get(ltype)
            if colour:
                out.append(f"  {a[colour]}{line}{a['reset']}")
            else:
                out.append(f"  {line}")

    return "\n".join(out)


def _display_full_raw_log(log_text: str) -> None:
    """Show the raw protocol log in a pager (``less``), one line per
    protocol message.  If stdout is not a terminal, print plainly."""
    import shutil
    import subprocess

    if not sys.stdout.isatty():
        print(log_text)
        return

    pager = os.environ.get("PAGER", "") or shutil.which("less") or ""
    if pager:
        cmd = [pager, "-RX", "-"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        try:
            proc.communicate(input=log_text, timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    else:
        print(log_text)


# ── Async helpers ────────────────────────────────────────────────────────────

async def _ainput(prompt: str = "", *, use_raw: bool = False) -> str:
    """Async wrapper around line input.

    When ``use_raw`` is True, reads via `_read_line_in_raw_mode` (needed
    when `_start_raw_mode` has been called).  Otherwise uses the standard
    ``input()`` function in a thread.
    """
    if use_raw:
        loop = asyncio.get_event_loop()
        return (await loop.run_in_executor(None, _read_line_in_raw_mode, prompt)).strip().lower()
    return (await asyncio.to_thread(input, prompt)).strip().lower()


# ── Main entry point ──────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Inspect a metamon replay at all pipeline stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step through a battle by its game ID
  python tools/inspect_replay.py gen4uu-184050323

  # Show only raw log (no forward/parsed)
  python tools/inspect_replay.py gen4uu-184050323 --raw-only

  # Show overall summary only, don't step through
  python tools/inspect_replay.py gen4uu-184050323 --summary
        """,
    )
    parser.add_argument("gameid", help="Battle ID (e.g. gen4uu-184050323)")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Show only the raw protocol log (no forward/parsed processing)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-page summary instead of stepping through",
    )
    parser.add_argument(
        "--showdown",
        action="store_true",
        help="Open the replay on replay.pokemonshowdown.com and sync turn navigation",
    )
    parser.add_argument(
        "--chrome-port",
        type=int,
        default=9222,
        help="Chrome remote debugging port (default: 9222)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Open replay in browser but don't sync terminal navigation (read-only)",
    )
    args = parser.parse_args()

    gameid = args.gameid

    # ── 1. Find files ─────────────────────────────────────────────────────
    raw_path = _find_raw_replay(gameid)
    parsed_paths = _find_parsed_replays(gameid)

    if raw_path is None and not parsed_paths:
        print(f"❌  No replay data found for battle '{gameid}'")
        print(f"   Checked: {_CACHE_DIR}/raw-replays/ and parsed-replays/")
        sys.exit(1)

    # ── 2. Load raw data ──────────────────────────────────────────────────
    raw_data = None
    raw_turns = []
    full_raw_log = ""  # cached for "R" command
    if raw_path:
        with open(raw_path) as f:
            raw_data = orjson.loads(f.read())
        full_raw_log = raw_data.get("log", "")
        # Parse log into turn groups
        raw_turns = _parse_raw_turns(full_raw_log)

    # ── 3. Load parsed data ───────────────────────────────────────────────
    parsed_data = {}
    for result_key, path in parsed_paths.items():
        parsed_data[result_key] = _load_parsed(path)

    # ── 4. Print file overview ────────────────────────────────────────────
    _print_header("FILE OVERVIEW")
    print(f"Battle ID:  {gameid}")
    if raw_data:
        print(f"Format:     {raw_data.get('format', '?')}")
        print(f"Players:    {raw_data['players'][0]} vs {raw_data['players'][1]}")
        print(f"Uploaded:   {datetime.fromtimestamp(int(raw_data.get('uploadtime', 0)))}")
        print(f"Raw replay: {raw_path}")
    else:
        print("Raw replay: NOT FOUND")

    for result_key, path in parsed_paths.items():
        pdata = parsed_data[result_key]
        # Extract POV player name from filename
        fname = os.path.basename(path).replace(".json.lz4", "")
        parts = fname.split("_")
        pov_player = parts[2] if len(parts) > 2 else "?"
        opponent = parts[4] if len(parts) > 4 else "?"
        print(f"Parsed {result_key:5s}: POV of {pov_player} vs {opponent} — {len(pdata['states'])} states → .../{os.path.relpath(path, _CACHE_DIR) if _CACHE_DIR else path}")

    # ── 4b. Set up Showdown bridge (if requested) ───────────────────────
    bridge = None
    if args.showdown:
        from showdown_bridge import ShowdownBridge, CDPConnectionError
        bridge = ShowdownBridge(args.gameid, port=args.chrome_port)
        try:
            await bridge.connect()
            await bridge.open_replay()
            print(f"\n  🌐 Showdown replay opened — {bridge.max_turn} turns (0–{bridge.max_turn})")
            # Warn if the local raw data has different players than the online replay
            if raw_data and bridge.remote_players:
                remote_p1, remote_p2 = bridge.remote_players
                local_players = raw_data.get("players", [])
                if len(local_players) == 2:
                    if local_players[0] != remote_p1 or local_players[1] != remote_p2:
                        print(f"  ℹ️  Player names differ (dataset uses anonymized names):")
                        print(f"       Local:  {local_players[0]} vs {local_players[1]}")
                        print(f"       Remote: {remote_p1} vs {remote_p2}")
        except CDPConnectionError as e:
            print(f"\n  ⚠️  Showdown bridge unavailable: {e}")
            print(f"     Terminal-only mode.")
            print(f"     Start Chrome: open -a 'Google Chrome' --args --remote-debugging-port={args.chrome_port}")
            bridge = None

    if args.raw_only:
        _print_raw_only(raw_data, raw_turns, gameid)
        if bridge and bridge.is_active:
            print("\n  (Browser tab left open — close it manually)")
            # Don't disconnect — leave the tab open for the user to browse
        return

    # ── 5. Run forward pass ───────────────────────────────────────────────
    forward_replay = None
    if raw_data:
        print("\n⏳  Running forward pass (reconstructing spectator POV)...")
        forward_replay = _run_forward(raw_data)
        print(f"    ✓  {len(forward_replay.turnlist)} turns reconstructed")

    # ── 6. Summary mode ───────────────────────────────────────────────────
    if args.summary:
        _print_summary(gameid, raw_data, raw_turns, forward_replay, parsed_data)
        if bridge and bridge.is_active:
            # Jump to last turn for a good summary view
            try:
                await bridge.goto_turn(bridge.max_turn or 0)
            except Exception:
                pass
            print("\n  (Browser tab left open at final turn — close it manually)")
        return

    # ── 7. Interactive turn-by-turn mode ──────────────────────────────────

    # Build turn-indexed data
    # Raw: dict turn_num -> lines (turn 0 = preamble)
    raw_by_turn: Dict[int, List[str]] = {tnum: lines for tnum, lines in raw_turns}

    # Forward: turnlist[0] is preamble (pre-battle), turnlist[1] = turn 1, etc.
    fwd_turns = forward_replay.turnlist if forward_replay else []
    # Build a dict turn_num -> Turn for easy lookup
    fwd_by_num: Dict[int, object] = {}
    for t in fwd_turns:
        if t.turn_number is not None:
            fwd_by_num[t.turn_number] = t

    # Parsed: flat sequence of decision points (not aligned to global turns)
    parsed_win = parsed_data.get("WIN")
    parsed_loss = parsed_data.get("LOSS")

    max_turns = max(len(raw_turns), max(fwd_by_num.keys()) if fwd_by_num else 0)
    n_parsed_states = max(
        len(parsed_win["states"]) if parsed_win else 0,
        len(parsed_loss["states"]) if parsed_loss else 0,
    )

    print(f"\n  📊 Raw turns: {len(raw_turns)}  |  Forward turns: {len(fwd_turns)}")
    print(f"  🤖 Parsed states: WIN={len(parsed_win['states']) if parsed_win else 0}"
          f"  LOSS={len(parsed_loss['states']) if parsed_loss else 0}")
    print(f"  ⚠️  Parsed states are per-player decision points (including sub-turns)")
    print(f"     and don't align 1:1 with global turns. Use 'a' to browse independently.")
    if bridge and bridge.is_active and not args.no_sync:
        print(f"  🌐 Browser sync ACTIVE — turns navigate together")
    elif bridge and bridge.is_active:
        print(f"  🌐 Browser OPEN (read-only, --no-sync)")
    print()
    print(f"  Press ENTER or → to advance, ← to go back, 'q' to quit.")
    print(f"  Commands: n/p/j<N>/r/R/f/a/s/h")

    current_turn = 1  # start at turn 1
    current_parsed = 0  # separate index for parsed state browsing

    # Switch terminal to raw mode for arrow-key support
    _start_raw_mode()

    async def _sync_browser(turn: int) -> None:
        """Send a turn change to the browser, swallowing errors."""
        if bridge and bridge.is_active and not args.no_sync:
            try:
                await bridge.goto_turn(turn)
            except Exception as e:
                print(f"  ⚠️  Browser sync failed: {e}")

    def _display_current_turn() -> None:
        """Print the full turn display (raw, forward, parsed) for `current_turn`."""
        _print_turn_separator(current_turn)
        # ── Raw log ──
        if current_turn in raw_by_turn:
            raw_lines = raw_by_turn[current_turn]
            print(f"\n  📝 RAW LOG ({len(raw_lines)} lines):")
            counts = _count_line_types(raw_lines)
            type_summary = " | ".join(f"{t}={c}" for t, c in sorted(counts.items()))
            print(f"     Types: {type_summary}")
            important = [l for l in raw_lines
                         if _raw_line_type(l) in ("move", "switch", "faint", "drag")]
            important += [l for l in raw_lines
                          if _raw_line_type(l) in ("-damage", "-heal", "-status",
                                                    "-boost", "-unboost", "-sidestart",
                                                    "-sideend", "-fieldstart", "-fieldend",
                                                    "-weather", "-ability", "-item", "-enditem",
                                                    "-curestatus", "-transform")]
            for line in (important[:10] if len(important) > 10 else raw_lines[:10]):
                print(f"     {line}")
            if len(important) > 10:
                print(f"     ... ({len(important) - 10} more important lines, use 'r' for full)")
        else:
            print("\n  📝 RAW LOG: (no raw data for this turn)")
        # ── Forward pass ──
        print(f"\n  🔍 FORWARD (spectator view):")
        if current_turn in fwd_by_num:
            for s in _forward_turn_summary(fwd_by_num[current_turn],
                                            forward_replay.gen if forward_replay else 1):
                print(s)
        else:
            print("    (no forward data for this turn)")
        # ── Parsed (approximate index) ──
        approx_parsed = int(current_turn * n_parsed_states / max(max_turns, 1)) if n_parsed_states else 0
        approx_parsed = min(approx_parsed, n_parsed_states - 1)
        print(f"\n  🤖 PARSED (RL training data, approx index {approx_parsed}):")
        print(f"     (parsed states are per-player decision points; use 'a' to browse accurately)")
        _display_parsed_at_index(parsed_win, parsed_loss, approx_parsed)

    # Show turn 1 immediately (before the first prompt)
    if bridge and bridge.is_active and not args.no_sync:
        try:
            await bridge.goto_turn(1)
        except Exception:
            pass
    _display_current_turn()

    while True:
        # Show which parsed index roughly corresponds to the current forward turn
        approx_parsed = int(current_turn * n_parsed_states / max(max_turns, 1)) if n_parsed_states else 0
        approx_parsed = min(approx_parsed, n_parsed_states - 1)

        # Poll for arrow keys first (non-blocking in raw mode)
        key = _poll_keypress()
        if key == "RIGHT" or key == "n":
            current_turn = min(current_turn + 1, max_turns)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue
        elif key == "LEFT" or key == "p":
            current_turn = max(current_turn - 1, 1)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue
        elif key == "ENTER":
            current_turn = min(current_turn + 1, max_turns)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue

        cmd = await _ainput(
            f"\n📌 [turn {current_turn}/{max_turns}]"
            f"  (parsed ~{approx_parsed}/{n_parsed_states-1}) > ",
            use_raw=True,
        )

        if cmd in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        elif cmd in ("n", "next"):
            current_turn = min(current_turn + 1, max_turns)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue
        elif cmd in ("p", "prev"):
            current_turn = max(current_turn - 1, 1)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue
        elif cmd in ("",):
            # Empty line (Enter) — same as next
            current_turn = min(current_turn + 1, max_turns)
            await _sync_browser(current_turn)
            _display_current_turn()
            continue
        elif cmd in ("h", "help"):
            print("  n/next/Enter/→ – next turn")
            print("  p/prev/←       – previous turn")
            print("  j <N>          – jump to turn N")
            print("  r              – show full raw log for current turn")
            print("  R              – show entire raw log (colourised, paged)")
            print("  f              – show full forward state for current turn")
            print("  a              – browse parsed states independently")
            print("  s              – show summary")
            print("  q/quit/exit    – quit")
            continue
        elif cmd.startswith("j"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                target = int(parts[1])
                if 1 <= target <= max_turns:
                    current_turn = target
                    await _sync_browser(current_turn)
                    _display_current_turn()
                else:
                    print(f"  Turn out of range (1–{max_turns})")
            else:
                print("  Usage: j <turn_number>")
                continue
        elif cmd.isdigit():
            target = int(cmd)
            if 1 <= target <= max_turns:
                current_turn = target
                await _sync_browser(current_turn)
                _display_current_turn()
            else:
                print(f"  Turn out of range (1–{max_turns})")
                continue
        elif cmd == "r":
            _display_raw_turn(raw_by_turn, current_turn)
            continue
        elif cmd == "R":
            _display_full_raw_log(full_raw_log)
            continue
        elif cmd == "f":
            _display_forward_turn_full(fwd_by_num, current_turn)
            continue
        elif cmd == "a":
            current_parsed = await _browse_parsed(parsed_win, parsed_loss, current_parsed)
            continue
        elif cmd == "s":
            _print_summary(gameid, raw_data, raw_turns, forward_replay, parsed_data)
            continue
        else:
            print(f"  Unknown command: '{cmd}' (type 'h' for help)")
            continue

    # ── Cleanup ───────────────────────────────────────────────────────────
    _stop_raw_mode()
    if bridge and bridge.is_active:
        try:
            await bridge.disconnect()
        except Exception:
            pass


def _display_raw_turn(raw_by_turn: Dict[int, List[str]], turn_num: int):
    """Display only raw log for a turn."""
    if turn_num in raw_by_turn:
        print(f"\n  📝 RAW LOG (turn {turn_num}):")
        for line in raw_by_turn[turn_num]:
            print(f"     {line}")
    else:
        print(f"  No raw data for turn {turn_num}")


def _display_forward_turn_full(fwd_by_num: Dict[int, object], turn_num: int):
    """Display detailed forward pass state for a turn."""
    if turn_num in fwd_by_num:
        turn = fwd_by_num[turn_num]
        gen = 1
        for p in turn.all_pokemon:
            if p:
                gen = p.gen
                break
        print(f"\n  🔍 FORWARD (spectator view, turn {turn_num}):")
        for s in _forward_turn_summary(turn, gen):
            print(s)
        # Show team state
        print(f"\n    Team state:")
        for tag, team in [("P1", turn.pokemon_1), ("P2", turn.pokemon_2)]:
            active_ids = {a.unique_id for a in (turn.active_pokemon_1 if tag == "P1" else turn.active_pokemon_2) if a}
            for p in team:
                if p is None:
                    continue
                marker = "⭐" if p.unique_id in active_ids else "  "
                hp = f"{p.current_hp}/{p.max_hp}" if p.max_hp else "?/?"
                status = str(p.status).split(".")[-1] if p.status else "none"
                print(f"      {marker} {tag}: {p.name:14s} HP={hp:>8s}  status={status}"
                      f"  item={str(p.active_item):15s}  ability={str(p.active_ability)}")
    else:
        print(f"  No forward data for turn {turn_num}")


def _display_parsed_at_index(parsed_win: dict, parsed_loss: dict, idx: int):
    """Display parsed state at a specific index (decision point)."""
    for label, pdata in [("WIN (P1 POV)", parsed_win), ("LOSS (P2 POV)", parsed_loss)]:
        if not pdata:
            print(f"    {label}: (no data)")
            continue
        if idx >= len(pdata["states"]):
            print(f"    {label}: index {idx} out of range ({len(pdata['states'])} states)")
            continue
        state = pdata["states"][idx]
        action_idx = pdata["actions"][idx] if idx < len(pdata["actions"]) else -1
        action_label = _action_to_label(state, action_idx)
        print(f"    {label}:")
        print(f"      Active:   {_pokemon_summary(state['player_active_pokemon'])}")
        print(f"      Opponent: {_pokemon_summary(state['opponent_active_pokemon'])}")
        print(f"      Switches: {len(state['available_switches'])} available")
        print(f"      Action:   {action_label}")
        print(f"      forced_switch={state['forced_switch']}"
              f" | won={state['battle_won']} | lost={state['battle_lost']}")


async def _browse_parsed(parsed_win: dict, parsed_loss: dict, start_idx: int = 0):
    """Interactive browser for parsed states (decision points)."""
    n_states = max(
        len(parsed_win["states"]) if parsed_win else 0,
        len(parsed_loss["states"]) if parsed_loss else 0,
    )
    if n_states == 0:
        print("  No parsed data available.")
        return start_idx

    idx = start_idx
    print(f"\n  🤖 PARSED STATE BROWSER ({n_states} states)")
    print(f"  n/Enter = next  |  p = prev  |  j<N> = jump  |  q = back to turns")

    while True:
        print(f"\n  ── Parsed state {idx}/{n_states-1} ──")
        _display_parsed_at_index(parsed_win, parsed_loss, idx)

        cmd = await _ainput(f"\n  📌 [parsed {idx}/{n_states-1}] > ")
        if cmd in ("q", "quit", "back", ""):
            break
        elif cmd in ("n", "next"):
            idx = min(idx + 1, n_states - 1)
        elif cmd in ("p", "prev"):
            idx = max(idx - 1, 0)
        elif cmd.isdigit():
            target = int(cmd)
            if 0 <= target < n_states:
                idx = target
            else:
                print(f"  Out of range (0–{n_states-1})")
        elif cmd.startswith("j"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                target = int(parts[1])
                if 0 <= target < n_states:
                    idx = target
                else:
                    print(f"  Out of range (0–{n_states-1})")

    return idx  # return current index for caller


# ── Summary & raw-only modes ──────────────────────────────────────────────────

def _print_raw_only(raw_data: dict, raw_turns: list, gameid: str):
    """Print the raw protocol log with metadata."""
    if not raw_data:
        print("No raw replay data found.")
        return

    _print_header("RAW REPLAY DETAIL")
    print(f"Game ID:    {gameid}")
    print(f"Format:     {raw_data.get('format', '?')}")
    print(f"Players:    {raw_data['players'][0]} vs {raw_data['players'][1]}")
    print(f"Uploaded:   {datetime.fromtimestamp(int(raw_data.get('uploadtime', 0)))}")
    print(f"Total log lines: {len(raw_data['log'].splitlines())}")
    print(f"Turns (including preamble): {len(raw_turns)}")

    # Line type histogram
    all_counts: Dict[str, int] = {}
    for _, lines in raw_turns:
        for t, c in _count_line_types(lines).items():
            all_counts[t] = all_counts.get(t, 0) + c
    print(f"\nProtocol line types:")
    for t, c in sorted(all_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {c:6d}")

    # Print first 3 turns
    for tnum, lines in raw_turns[:4]:
        _print_turn_separator(tnum)
        for line in lines[:15]:
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... ({len(lines) - 15} more lines)")


def _print_summary(gameid: str, raw_data: dict, raw_turns: list,
                   forward_replay, parsed_data: dict):
    """Print a one-page battle summary."""
    _print_header("BATTLE SUMMARY")

    # Metadata
    if raw_data:
        print(f"  Format:   {raw_data.get('format', '?')}")
        print(f"  Players:  {raw_data['players'][0]} vs {raw_data['players'][1]}")
        print(f"  Date:     {datetime.fromtimestamp(int(raw_data.get('uploadtime', 0)))}")

    # Turn counts from each source
    print(f"\n  Raw turns:      {len(raw_turns)}")
    if forward_replay:
        print(f"  Forward turns:  {len(forward_replay.turnlist)}")
        # Show final team state
        print(f"\n  Final teams (from forward pass):")
        if forward_replay.turnlist:
            last_turn = forward_replay.turnlist[-1]
            for tag, team in [("P1", last_turn.pokemon_1), ("P2", last_turn.pokemon_2)]:
                print(f"    {tag}:")
                for p in team:
                    if p is None:
                        continue
                    hp_str = f"{p.current_hp}/{p.max_hp}" if p.max_hp else "?"
                    moves = ", ".join(m.name for m in p.moves.values())
                    status = str(p.status).split(".")[-1] if p.status else "none"
                    print(f"      {p.name:15s} HP={hp_str:>8s}  item={p.had_item or '?'}"
                          f"  ability={p.had_ability or '?'}  status={status}")
                    print(f"        moves: {moves}" if moves else "        moves: (none)")

    for result_key, pdata in parsed_data.items():
        n_states = len(pdata["states"])
        n_actions = len(pdata["actions"])
        print(f"\n  Parsed {result_key}: {n_states} states, {n_actions} actions")
        # Show first and last state summary
        if n_states > 0:
            first = pdata["states"][0]
            last = pdata["states"][-1]
            print(f"    First active: {_pokemon_summary(first['player_active_pokemon'])}")
            print(f"    Final result: won={last['battle_won']}, lost={last['battle_lost']}")
            # Action distribution
            from collections import Counter
            action_counts = Counter(pdata["actions"])
            print(f"    Action distribution: {dict(sorted(action_counts.items()))}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _stop_raw_mode()
        print("\nInterrupted.")
    except EOFError:
        _stop_raw_mode()
        print("\nGoodbye!")
