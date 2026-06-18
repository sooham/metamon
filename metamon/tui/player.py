"""Model‑agnostic TUI mixin for Metamon players.

Adds an interactive full‑screen REPL (R=raw, P=blocks, V=verbose,
O=overview, Q=quit, ↑↓ navigate, Enter inspect) on top of any
MetamonPlayer subclass.  The concrete player must set:

* ``self._tokenizer`` — a tokenizer with ``detokenize`` and ``pad_token_id``
* ``self._histories: dict[str, BattleHistory]`` — per‑battle data store
* ``self._battles`` — dict of active battles (from MetamonPlayer)
* ``self._last_active_battle_tag: str | None``
* ``self._tui_detok(tokens) -> str`` — detokenize a numpy array to text
* ``self._tui_store_turn(diag, action_names)`` — called after every
  ``choose_move`` decision; appends to ``diag_history`` etc.
* ``self._tui_current_hist() -> BattleHistory | None`` — return the
  ``BattleHistory`` for the most recently active battle.
"""

from __future__ import annotations

import os as _os
import select as _select
import sys as _sys
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from metamon.tui.theme import (
    TURN_BG_COLORS,
    colorize_raw_line,
    colorize_state_line,
)


# ── tiny display helpers ──────────────────────────────────────────────

def _display_hp(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{max(0.0, min(1.0, float(value))):.2f}"


def _display_species(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    return str(getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None) or "unknown")


# ── TUI–model contract ──────────────────────────────────────────────

@dataclass
class TurnContext:
    """Battle metadata filled by the *player* (not the model) each turn."""
    battle_tag: str
    turn: int
    opponent_name: str
    active_species: str
    active_hp: str
    opponent_species: str
    opponent_hp: str
    heuristic: str
    forced_switch: bool = False
    num_state_blocks: int = 0
    num_player_actions: int = 0
    num_opponent_actions: int = 0


@dataclass
class TuiDiagnostic:
    """Per‑turn diagnostic data produced by the *model*.

    The TUI renders this without knowing anything about the model's
    internals — it just displays the declared columns and rows.
    """
    # Column headers for the per‑action table (e.g. ["action", "name", "rank", …]).
    columns: list[str]
    # One row per legal action; each row is a list of strings (one per column).
    rows: list[list[str]]
    # Index of the action the model chose (points into ``rows``).
    chosen_idx: int
    # Compact label for the chosen action (e.g. "move: thunderwave").
    chosen_label: str
    # Optional free‑form lines shown above the table (model‑specific context).
    context_lines: list[str] = field(default_factory=list)


# ── BattleHistory (model‑agnostic per‑battle data) ───────────────────

@dataclass
class BattleHistory:
    """Per-battle state tracked across turns for model input and REPL display."""

    state_blocks: list[np.ndarray] = field(default_factory=list)
    player_actions: list[np.ndarray] = field(default_factory=list)
    opponent_actions: list[np.ndarray] = field(default_factory=list)
    pending_player_action: Optional[np.ndarray] = None
    pending_player_action_text: Optional[str] = None
    pending_forced_switch: bool = False
    last_state_key: Optional[tuple[int, ...]] = None
    current_forced_switch: bool = False
    # Raw protocol messages (pipe-delimited strings) for the "R" REPL key.
    raw_messages: list[str] = field(default_factory=list)
    # Per-turn diagnostics history for REPL views.
    turn_contexts: list[TurnContext] = field(default_factory=list)
    turn_diagnostics: list[TuiDiagnostic] = field(default_factory=list)
    # Indices into raw_messages where each turn started (for per‑turn raw log view).
    turn_msg_boundaries: list[int] = field(default_factory=list)
    # Battle outcome tracking.
    finished: bool = False
    outcome: Optional[str] = None  # "win", "loss", "tie", or None


# ═════════════════════════════════════════════════════════════════════════
# TuiMixin
# ═════════════════════════════════════════════════════════════════════════

class TuiMixin:
    """Mixin that adds a full‑screen REPL to a MetamonPlayer.

    Concrete players must override the ``_tui_*`` hook methods below."""

    # ── Shared REPL state (class-level, one key listener for all instances) ──
    _repl_keys: list[str] = []
    _repl_running: bool = False
    _repl_lock = None
    _repl_view: str = "live"
    _repl_term = None
    _repl_active_instance: Optional["TuiMixin"] = None
    _repl_all_instances: list["TuiMixin"] = []
    _repl_verbose_blocks: bool = False
    _repl_last_auto_redraw: float = 0.0
    _repl_selected_turn: int = -1
    _repl_show_raw_for_turn: bool = False
    _repl_scroll_offset: int = 0
    _repl_save_raw_dir: Optional[str] = None  # set by play.py --save-raw-replay

    # ── hooks that concrete players must provide ──────────────────────

    def _tui_detok(self, tokens: np.ndarray) -> str:
        raise NotImplementedError("subclass must provide _tui_detok")

    def _tui_store_turn(self, ctx: TurnContext, diag: TuiDiagnostic) -> None:
        """Store a turn's context and diagnostics into the active BattleHistory."""
        raise NotImplementedError("subclass must provide _tui_store_turn")

    def _tui_current_hist(self) -> Optional[BattleHistory]:
        raise NotImplementedError("subclass must provide _tui_current_hist")

    # ── registration (called from concrete __init__) ──────────────────

    def _tui_register(self) -> None:
        """Register this instance for REPL routing."""
        TuiMixin._repl_active_instance = self
        TuiMixin._repl_all_instances.append(self)

    # ═══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def _start_repl(cls) -> None:
        if TuiMixin._repl_running:
            return
        TuiMixin._repl_running = True
        TuiMixin._repl_lock = threading.Lock()
        TuiMixin._repl_view = "live"
        from blessed import Terminal
        TuiMixin._repl_term = Terminal()
        _sys.stdout.write((TuiMixin._repl_term.enter_fullscreen() or "") + "\n")
        _sys.stdout.flush()
        t = threading.Thread(target=cls._key_listener, daemon=True)
        t.start()
        print("\n[JEPA REPL]  R=raw  P=blocks  V=verbose  O=overview  Q=quit")

    @classmethod
    def _stop_repl(cls) -> None:
        TuiMixin._repl_running = False
        if TuiMixin._repl_term is not None:
            _sys.stdout.write((TuiMixin._repl_term.exit_fullscreen() or "") + "\n")
            _sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════
    # Key listener (daemon thread)
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def _key_listener(cls) -> None:
        term = TuiMixin._repl_term
        fd = _sys.stdin.fileno()
        _sys.stderr.write(f"[TUI] listener started, fd={fd}, term={'OK' if term else 'None'}\n")
        _sys.stderr.flush()
        try:
            with term.cbreak():
                _sys.stderr.write("[TUI] cbreak entered\n")
                _sys.stderr.flush()
                while TuiMixin._repl_running:
                    try:
                        r, _, _ = _select.select([fd], [], [], 0.2)
                    except (InterruptedError, KeyboardInterrupt):
                        TuiMixin._repl_running = False
                        with TuiMixin._repl_lock:
                            TuiMixin._repl_keys.append("\x03")
                        break
                    except OSError:
                        TuiMixin._repl_running = False
                        break
                    if r:
                        data = _os.read(fd, 1)
                        if data:
                            low = data.decode("utf-8", errors="replace").lower()
                            _sys.stderr.write(f"[TUI] key: {repr(low)}\n")
                            _sys.stderr.flush()
                        if not data:
                            continue
                        key_str = data.decode("utf-8", errors="replace")
                        if key_str == "\x03":
                            TuiMixin._repl_running = False
                            with TuiMixin._repl_lock:
                                TuiMixin._repl_keys.append(key_str)
                            break
                        if key_str in ("\t", " "):
                            continue
                        # ── vim‑style navigation (j=down, k=up, h=up, g=top, G=bottom) ──
                        if key_str.lower() in ('j', 'k', 'h', 'g', 'l') or key_str == 'G':
                            self._tui_handle_nav_key(key_str)
                            continue
                        # Enter / Return
                        if key_str in ("\r", "\n"):
                            inst = TuiMixin._repl_active_instance
                            if inst is None or inst._last_active_battle_tag is None:
                                for c in TuiMixin._repl_all_instances:
                                    if c._last_active_battle_tag is not None:
                                        inst = c; break
                            if inst is not None:
                                hist = inst._tui_current_hist()
                                if hist is not None and hist.turn_contexts:
                                    TuiMixin._repl_show_raw_for_turn = not TuiMixin._repl_show_raw_for_turn
                                    inst._tui_redraw()
                            continue
                        # Escape — arrow sequences
                        if key_str == "\x1b":
                            r2, _, _ = _select.select([fd], [], [], 0.05)
                            if r2:
                                byte2 = _os.read(fd, 1)
                                if byte2 == b"[":
                                    r3, _, _ = _select.select([fd], [], [], 0.05)
                                    if r3:
                                        byte3 = _os.read(fd, 1)
                                        if byte3 == b"A":
                                            TuiMixin._tui_handle_nav_key("k")  # up
                                        elif byte3 == b"B":
                                            TuiMixin._tui_handle_nav_key("j")  # down
                            continue
                        # View‑switch keys (R, P, V, O)
                        low = key_str.lower()
                        if low in ('r', 'p', 'v', 'o'):
                            if low == 'r':
                                TuiMixin._repl_view = "raw"
                                TuiMixin._repl_show_raw_for_turn = False
                                TuiMixin._repl_scroll_offset = 0
                            elif low == 'p':
                                TuiMixin._repl_view = "blocks"
                                TuiMixin._repl_show_raw_for_turn = False
                                TuiMixin._repl_scroll_offset = 0
                            elif low == 'v':
                                TuiMixin._repl_verbose_blocks = not TuiMixin._repl_verbose_blocks
                                TuiMixin._repl_view = "live"
                                TuiMixin._repl_show_raw_for_turn = False
                                TuiMixin._repl_scroll_offset = 0
                            elif low == 'o':
                                TuiMixin._repl_view = "overview"
                                TuiMixin._repl_show_raw_for_turn = False
                                TuiMixin._repl_scroll_offset = 0
                            inst = TuiMixin._repl_active_instance
                            if inst is None or inst._last_active_battle_tag is None:
                                for c in TuiMixin._repl_all_instances:
                                    if c._last_active_battle_tag is not None:
                                        inst = c; break
                            if inst is not None:
                                inst._tui_redraw()
        except (KeyboardInterrupt, OSError):
            TuiMixin._repl_running = False
            with TuiMixin._repl_lock:
                TuiMixin._repl_keys.append("\x03")

    # ═══════════════════════════════════════════════════════════════════
    # Key processing (called from main event loop and listener)
    # ═══════════════════════════════════════════════════════════════════

    def _process_repl_keys(self) -> None:
        while True:
            with self._repl_lock:
                if not self._repl_keys:
                    break
                key = self._repl_keys.pop(0)
            if key == "\x03" or key.lower() == 'q':
                self._tui_clear()
                print("[REPL quit — restart with TuiMixin._start_repl()]")
                TuiMixin._repl_running = False
            elif key.lower() == 'r':
                TuiMixin._repl_view = "raw"
                TuiMixin._repl_scroll_offset = 0
                self._tui_redraw()
            elif key.lower() == 'p':
                TuiMixin._repl_view = "blocks"
                TuiMixin._repl_scroll_offset = 0
                self._tui_redraw()
            elif key.lower() == 'v':
                TuiMixin._repl_verbose_blocks = not TuiMixin._repl_verbose_blocks
                TuiMixin._repl_view = "live"
                TuiMixin._repl_show_raw_for_turn = False
                TuiMixin._repl_scroll_offset = 0
                self._tui_redraw()
            elif key.lower() == 'o':
                TuiMixin._repl_view = "overview"
                self._tui_redraw()

    # ═══════════════════════════════════════════════════════════════════
    # Screen management
    # ═══════════════════════════════════════════════════════════════════

    def _tui_clear(self) -> None:
        term = TuiMixin._repl_term
        if term is not None:
            # Write clear + home, then a newline so pipe consumers (tee) flush.
            _sys.stdout.write(term.clear + term.move_xy(0, 0) + "\n")
            _sys.stdout.flush()

    def _tui_redraw(self) -> None:
        import sys as _sys
        term = TuiMixin._repl_term
        if term is None:
            _sys.stderr.write("[TUI] _tui_redraw SKIP — term is None\n")
            _sys.stderr.flush()
            return
        view = TuiMixin._repl_view
        _sys.stderr.write(f"[TUI] _tui_redraw START view={view} term.width={term.width}\n")
        _sys.stderr.flush()
        width = term.width

        self._tui_clear()

        # ── header ──
        print(f"{'─' * width}")
        if view == "live":
            print(f"  JEPA REPL · live mode · verbose_blocks: {'ON' if TuiMixin._repl_verbose_blocks else 'OFF'}")
        elif view == "raw":
            tag = getattr(self, "_last_active_battle_tag", None)
            print(f"  JEPA REPL · raw protocol ({tag or 'no battle'})")
        elif view == "blocks":
            tag = getattr(self, "_last_active_battle_tag", None)
            print(f"  JEPA REPL · state / action blocks ({tag or 'no battle'})")
        elif view == "overview":
            print("  JEPA REPL · battle overview")
        print(f"{'─' * width}")

        # ── body ──
        if view == "live":
            self._tui_render_live()
        elif view == "raw":
            self._tui_render_raw()
        elif view == "blocks":
            self._tui_render_blocks()
        elif view == "overview":
            self._tui_render_overview()

        # ── footer ──
        print()
        print(f"{'─' * width}")
        footer_parts = [
            "[R]aw", "[P]arse",
            f"[V]erbose ({'ON' if TuiMixin._repl_verbose_blocks else 'OFF'})",
            "[O]verview", "[Q]uit",
        ]
        if view == "live":
            footer_parts.append("↑↓/jk nav" if not TuiMixin._repl_show_raw_for_turn else "ENTER back")
            if TuiMixin._repl_show_raw_for_turn:
                footer_parts.append("[Enter] turn detail")
        print(f"  {'  '.join(footer_parts)}")
        _sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════
    # View renderers
    # ═══════════════════════════════════════════════════════════════════

    def _tui_render_live(self) -> None:
        """Render the live view using TurnContext + TuiDiagnostic."""
        import sys as _sys
        hist = self._tui_current_hist()
        _sys.stderr.write(f"[TUI] _tui_render_live hist={hist is not None} turns={len(hist.turn_diagnostics) if hist else 0}\n")
        _sys.stderr.flush()
        term = TuiMixin._repl_term
        if hist is None:
            print("\n  waiting for battle...")
            return
        if not hist.turn_contexts:
            print(f"\n  states={len(hist.state_blocks)}  player_actions={len(hist.player_actions)}  opponent_actions={len(hist.opponent_actions)}")
            print("  waiting for first turn...")
            return

        nc = term.normal if term else ""
        verbose = TuiMixin._repl_verbose_blocks
        sel = TuiMixin._repl_selected_turn
        n_turns = len(hist.turn_contexts)
        if sel < 0 or sel >= n_turns:
            sel = n_turns - 1
            TuiMixin._repl_selected_turn = sel

        # ── compact turn list ──
        sel_bg = term.on_color_rgb(80, 80, 40) if term else ""
        dim = term.dim if term else ""
        for t_idx in range(n_turns):
            ctx = hist.turn_contexts[t_idx]
            diag = hist.turn_diagnostics[t_idx]
            marker = ">>>" if t_idx == sel else "   "
            line = (f"  {marker} turn {ctx.turn} ({ctx.battle_tag})"
                    f" [{ctx.heuristic}] — {diag.chosen_label}")
            if t_idx == sel:
                print(f"{sel_bg}{line}{nc}")
            else:
                print(f"{dim}{line}{nc}")

        # ── expanded selected turn ──
        ctx = hist.turn_contexts[sel]
        diag = hist.turn_diagnostics[sel]

        print(f"\n  ── turn {ctx.turn} ({ctx.battle_tag}) [{ctx.heuristic}] ──")
        print(f"    vs: {ctx.opponent_name}")
        print(f"    active: {ctx.active_species} hp={ctx.active_hp}")
        print(f"    opponent: {ctx.opponent_species} hp={ctx.opponent_hp}")
        if ctx.forced_switch:
            print("    state: forceswitch")

        if TuiMixin._repl_show_raw_for_turn:
            start = hist.turn_msg_boundaries[sel - 1] if sel > 0 else 0
            end = hist.turn_msg_boundaries[sel] if sel < len(hist.turn_msg_boundaries) else len(hist.raw_messages)
            total_msgs = end - start
            offset = TuiMixin._repl_scroll_offset
            visible = max(8, (term.height if term else 24) - 8)
            if offset + visible > total_msgs:
                offset = max(0, total_msgs - visible)
            if offset < 0:
                offset = 0
            TuiMixin._repl_scroll_offset = offset
            print(f"\n    raw log ({total_msgs} messages, turn boundary {start}–{end - 1})")
            if offset > 0:
                print(f"      ... ({offset} lines above) ...")
            shown = 0
            palette = TURN_BG_COLORS
            turn_bg_idx = 0
            current_bg = ""
            nc2 = term.normal if term else ""
            for line in hist.raw_messages[start + offset:end]:
                if shown >= visible:
                    break
                shown += 1
                if line.startswith("|turn|"):
                    r, g, b = palette[turn_bg_idx % len(palette)]
                    current_bg = term.on_color_rgb(r, g, b) if term else ""
                    turn_bg_idx += 1
                colored = colorize_raw_line(line, term=term, turn_bg=current_bg)
                print(f"      {colored}")
            if offset + visible < total_msgs:
                print(f"      ... ({total_msgs - offset - visible} lines below) ...")
        elif verbose:
            n_states = ctx.num_state_blocks
            n_player = ctx.num_player_actions
            n_opp = ctx.num_opponent_actions
            print(f"    blocks: states={n_states} player_actions={n_player} opponent_actions={n_opp}")
            # Show team header (first block) for context.
            if len(hist.state_blocks) >= 1:
                team = hist.state_blocks[0]
                colored_team = colorize_state_line(self._tui_detok(team), term=term)
                print(f"    [team] {colored_team}")
            # Show the most recent state block.
            if len(hist.state_blocks) >= 2:
                last_block = hist.state_blocks[-1]
                label = f"state_{len(hist.state_blocks) - 2}"
                colored = colorize_state_line(self._tui_detok(last_block), term=term)
                print(f"    [{label}] {colored}")
            # Model‑specific context lines.
            for line in diag.context_lines:
                print(f"    {line}")
            # Per‑action table.
            if diag.rows:
                header = "    " + "  ".join(f"{c:>{max(len(c), 8)}}" for c in diag.columns)
                print(header)
                for r_idx, row in enumerate(diag.rows):
                    marker = " <- chosen" if r_idx == diag.chosen_idx else ""
                    formatted = "    " + "  ".join(
                        f"{v:>{max(len(diag.columns[i]), 8)}}"
                        for i, v in enumerate(row)
                    )
                    print(f"{formatted}{marker}")
        else:
            print(f"    chosen: {diag.chosen_label}")

    def _tui_render_raw(self) -> None:
        hist = self._tui_current_hist()
        if hist is None:
            print("\n  [no active battle]")
            return
        total = len(hist.raw_messages)
        offset = TuiMixin._repl_scroll_offset
        term = TuiMixin._repl_term
        visible = max(10, (term.height if term else 24) - 5)
        if offset + visible > total:
            offset = max(0, total - visible)
        TuiMixin._repl_scroll_offset = offset
        print(f"  {total} total messages  (lines {offset + 1}–{min(total, offset + visible)})")
        if offset > 0:
            print(f"  ... ({offset} lines above) ...")
        palette = TURN_BG_COLORS
        turn_idx = 0
        current_bg = ""
        shown = 0
        for line in hist.raw_messages[offset:]:
            if shown >= visible:
                break
            shown += 1
            if line.startswith("|turn|"):
                r, g, b = palette[turn_idx % len(palette)]
                current_bg = term.on_color_rgb(r, g, b) if term else ""
                turn_idx += 1
            colored = colorize_raw_line(line, term=term, turn_bg=current_bg)
            print(f"  {colored}")
        if offset + visible < total:
            print(f"  ... ({total - offset - visible} lines below) ...")

    def _tui_render_blocks(self) -> None:
        hist = self._tui_current_hist()
        if hist is None:
            print("\n  [no active battle]")
            return
        print(f"  states={len(hist.state_blocks)}  player_actions={len(hist.player_actions)}  opponent_actions={len(hist.opponent_actions)}")
        print()
        term = TuiMixin._repl_term
        for i, block in enumerate(hist.state_blocks):
            label = "team_header" if i == 0 else f"state_{i - 1}"
            colored = colorize_state_line(self._tui_detok(block), term=term)
            print(f"  [{label}] {colored}")
            if i >= 1:
                ai = i - 1
                if ai < len(hist.player_actions):
                    pa_colored = colorize_state_line(self._tui_detok(hist.player_actions[ai]), term=term)
                    print(f"  [player_action_{ai}] {pa_colored}")
                if ai < len(hist.opponent_actions):
                    oa_colored = colorize_state_line(self._tui_detok(hist.opponent_actions[ai]), term=term)
                    print(f"  [opponent_action_{ai}] {oa_colored}")
        for turn_idx in range(len(hist.turn_contexts)):
            ctx = hist.turn_contexts[turn_idx]
            diag = hist.turn_diagnostics[turn_idx]
            print(f"\n  ── turn {ctx.turn} diagnostics ──")
            for line in diag.context_lines:
                print(f"    {line}")

    def _tui_render_overview(self) -> None:
        any_battle = False
        for inst in TuiMixin._repl_all_instances:
            histories = getattr(inst, "_histories", {})
            battles = getattr(inst, "_battles", {})
            for tag, hist in histories.items():
                any_battle = True
                n_states = len(hist.state_blocks)
                n_player = len(hist.player_actions)
                n_opp = len(hist.opponent_actions)
                n_raw = len(hist.raw_messages)
                extra = ""
                if tag in battles:
                    b = battles[tag]
                    our_species = _display_species(b.active_pokemon)
                    our_hp = _display_hp(getattr(b.active_pokemon, "current_hp_fraction", None))
                    opp_species = _display_species(b.opponent_active_pokemon)
                    opp_hp = _display_hp(getattr(b.opponent_active_pokemon, "current_hp_fraction", None))
                    extra = f"  {our_species} hp={our_hp} vs {opp_species} hp={opp_hp}"
                    # Show our full team.
                    our_team = [
                        getattr(p, "species", None) or getattr(p, "base_species", None) or "?"
                        for p in b.team.values()
                    ]
                    if our_team:
                        extra += f"\n    team: {', '.join(our_team)}"
                print(f"  {tag}: turn≈{n_states - 1}  states={n_states}  actions=({n_player},{n_opp})  msgs={n_raw}{extra}")
                # Show outcome for finished battles.
                if hist.finished:
                    outcomes = {"win": "[W]", "loss": "[L]", "tie": "[T]"}
                    marker = outcomes.get(hist.outcome, "[?]")
                    dim = TuiMixin._repl_term.dim if TuiMixin._repl_term else ""
                    nc = TuiMixin._repl_term.normal if TuiMixin._repl_term else ""
                    print(f"    {dim}{marker} finished ({hist.outcome}){nc}")
        if not any_battle:
            print("\n  [no active battles]")

    # ═══════════════════════════════════════════════════════════════════
    # Protocol‑message capture (override _handle_battle_message)
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_battle_message(self, split_messages):
        """Capture raw messages + drain REPL keys + auto‑refresh views."""
        import sys as _sys
        try:
            battle_tag = self._normalise_tag(split_messages[0][0])
            _sys.stderr.write(f"[TUI] hbm {battle_tag} msgs={len(split_messages)-1}\n")
            _sys.stderr.flush()
            hist = self._histories.get(battle_tag)
            if hist is None:
                hist = BattleHistory()
                self._histories[battle_tag] = hist
                _sys.stderr.write(f"[TUI] new history for {battle_tag}\n")
                _sys.stderr.flush()
            for msg in split_messages[1:]:
                if len(msg) > 1:
                    hist.raw_messages.append("|".join(msg))
                    # Detect battle outcomes.
                    if msg[0] == "win":
                        hist.finished = True
                        winner = msg[1] if len(msg) > 1 else ""
                        our_name = getattr(self, "username", "")
                        hist.outcome = "win" if winner == our_name else "loss"
                    elif msg[0] == "tie":
                        hist.finished = True
                        hist.outcome = "tie"
                    # Save raw replay when battle finishes.
                    if msg[0] in ("win", "tie") and TuiMixin._repl_save_raw_dir:
                        self._tui_save_raw_replay(battle_tag, hist)
            self._last_active_battle_tag = battle_tag
            TuiMixin._repl_active_instance = self
            self._process_repl_keys()
            # Auto‑refresh the current view on every protocol message (throttled).
            now = _time.monotonic()
            if now - TuiMixin._repl_last_auto_redraw > 0.1:
                TuiMixin._repl_last_auto_redraw = now
                self._tui_redraw()
            # If the current battle just finished, switch to another live one
            # or fall back to the overview.
            if hist.finished and TuiMixin._repl_view == "live":
                other = self._tui_find_live_battle()
                if other is not None and other is not self:
                    TuiMixin._repl_active_instance = other
                    self._last_active_battle_tag = other._last_active_battle_tag
                    other._tui_redraw()
                elif other is self:
                    # Current battle already redirected, or single battle still going.
                    self._tui_redraw()
                else:
                    # No live battles — show overview.
                    TuiMixin._repl_view = "overview"
                    self._tui_redraw()
        except Exception as e:
            _sys.stderr.write(f"[TUI] _handle_battle_message error: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc(file=_sys.stderr)
            _sys.stderr.flush()
        # Always delegate to the concrete player's parent.
        await super()._handle_battle_message(split_messages)  # type: ignore[misc]

    @staticmethod
    def _normalise_tag(raw_tag: str) -> str:
        """Strip the leading '>' from protocol room identifiers."""
        return raw_tag[1:] if raw_tag.startswith(">") else raw_tag

    @classmethod
    def _tui_handle_nav_key(cls, key_str: str) -> None:
        """Handle a vim‑style navigation key (j, k, h, g, G, l) or arrow key.

        Called from the listener thread; resolves the best instance and
        applies the navigation action based on the current view."""
        inst = cls._repl_active_instance
        if inst is None or inst._last_active_battle_tag is None:
            for c in cls._repl_all_instances:
                if c._last_active_battle_tag is not None:
                    inst = c; break
        if inst is None:
            return
        hist = inst._tui_current_hist()
        if hist is None:
            return
        low = key_str.lower()
        view = cls._repl_view
        # Determine direction: up / down / top / bottom
        if low in ('k', 'h'):
            direction = "up"
        elif low in ('j', 'l'):
            direction = "down"
        elif low == 'g':
            direction = "top"
        elif key_str == 'G':  # uppercase only
            direction = "bottom"
        else:
            return

        if view == "raw":
            total = len(hist.raw_messages)
            if direction == "up":
                cls._repl_scroll_offset = max(0, cls._repl_scroll_offset - 1)
            elif direction == "down":
                cls._repl_scroll_offset = min(max(0, total - 1), cls._repl_scroll_offset + 1)
            elif direction == "top":
                cls._repl_scroll_offset = 0
            elif direction == "bottom":
                cls._repl_scroll_offset = max(0, total - 1)
            inst._tui_redraw()
        elif view == "live" and cls._repl_show_raw_for_turn:
            if cls._repl_selected_turn >= 0 and cls._repl_selected_turn < len(hist.turn_contexts):
                start = hist.turn_msg_boundaries[cls._repl_selected_turn - 1] if cls._repl_selected_turn > 0 else 0
                end = hist.turn_msg_boundaries[cls._repl_selected_turn] if cls._repl_selected_turn < len(hist.turn_msg_boundaries) else len(hist.raw_messages)
                total = max(1, end - start)
                if direction == "up":
                    cls._repl_scroll_offset = max(0, cls._repl_scroll_offset - 1)
                elif direction == "down":
                    cls._repl_scroll_offset = min(total - 1, cls._repl_scroll_offset + 1)
                elif direction == "top":
                    cls._repl_scroll_offset = 0
                elif direction == "bottom":
                    cls._repl_scroll_offset = max(0, total - 1)
                inst._tui_redraw()
        elif view == "live" and hist.turn_contexts:
            n = len(hist.turn_contexts)
            if direction == "up":
                cls._repl_selected_turn = max(0, cls._repl_selected_turn - 1)
            elif direction == "down":
                cls._repl_selected_turn = min(n - 1, cls._repl_selected_turn + 1)
            elif direction == "top":
                cls._repl_selected_turn = 0
            elif direction == "bottom":
                cls._repl_selected_turn = n - 1
            inst._tui_redraw()
        # Blocks view — scroll through the block listing.
        elif view == "blocks":
            total = len(hist.state_blocks) + len(hist.player_actions) + len(hist.opponent_actions)
            if direction == "up":
                cls._repl_scroll_offset = max(0, cls._repl_scroll_offset - 1)
            elif direction == "down":
                cls._repl_scroll_offset = min(max(0, total - 1), cls._repl_scroll_offset + 1)
            elif direction == "top":
                cls._repl_scroll_offset = 0
            elif direction == "bottom":
                cls._repl_scroll_offset = max(0, total - 1)
            inst._tui_redraw()

    @classmethod
    def _tui_find_live_battle(cls) -> Optional["TuiMixin"]:
        """Return an instance that has an unfinished battle, or None."""
        for inst in cls._repl_all_instances:
            tag = inst._last_active_battle_tag
            if tag is None:
                continue
            hist = inst._histories.get(tag)
            if hist is not None and not hist.finished:
                return inst
        return None

    def _tui_save_raw_replay(self, battle_tag: str, hist: BattleHistory) -> None:
        """Save the raw protocol messages for a finished battle to disk.

        Filename follows the convention:
          {format}-{battle_id}_{player}_vs_{opponent}_{date}_{OUTCOME}.txt
        """
        save_dir = TuiMixin._repl_save_raw_dir
        if not save_dir or not hist.raw_messages:
            return
        import os as _os2
        from datetime import datetime

        # Extract the numeric battle ID from the tag (e.g. "battle-gen1ou-55" → "55").
        battle_id = battle_tag.rsplit("-", 1)[-1]
        # Extract format from the tag (e.g. "battle-gen1ou-55" → "gen1ou").
        tag_format = battle_tag.split("-", 1)[1] if "-" in battle_tag else battle_tag
        tag_format = tag_format.rsplit("-", 1)[0] if "-" in tag_format else tag_format

        # Try to find opponent name from |player| messages.
        our_name = getattr(self, "username", "player")
        opponent = "unknown"
        for msg in hist.raw_messages:
            if msg.startswith("|player|"):
                parts = msg.split("|")
                if len(parts) >= 3 and parts[2] != our_name:
                    opponent = parts[2]
                    break

        date_str = datetime.now().strftime("%m-%d-%Y")
        outcome_str = hist.outcome.upper() if hist.outcome else "UNKNOWN"

        filename = (
            f"{tag_format}-{battle_id}_{our_name}_vs_{opponent}"
            f"_{date_str}_{outcome_str}.txt"
        )
        path = _os2.path.join(save_dir, filename)
        try:
            header = (
                f"# outcome: {hist.outcome or 'unknown'}\n"
                f"# messages: {len(hist.raw_messages)}\n"
                f"# tag: {battle_tag}\n\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write("\n".join(hist.raw_messages))
                f.write("\n")
        except OSError:
            pass  # silently ignore write errors
