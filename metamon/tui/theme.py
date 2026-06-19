"""ANSI colour helpers for the JEPA REPL TUI.

Kept in a separate module so player.py stays focused on battle logic.
All functions take an optional *term* (blessed Terminal) and degrade
gracefully when it is None (no colour output).
"""

from __future__ import annotations

from typing import Optional

# ── turn background palette (cycled) ────────────────────────────────

TURN_BG_COLORS: list[tuple[int, int, int]] = [
    (48, 48, 64),    # dark blue‑grey
    (48, 64, 48),    # dark green‑grey
    (64, 48, 48),    # dark red‑grey
    (48, 48, 80),    # dark periwinkle
    (64, 56, 48),    # dark amber
    (48, 64, 64),    # dark teal
    (56, 48, 64),    # dark plum
    (64, 64, 48),    # dark olive
]

# ── Pokémon type colours ─────────────────────────────────────────────

_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "normal":   (168, 168, 120),
    "fire":     (240, 128, 48),
    "water":    (104, 144, 240),
    "electric": (248, 208, 48),
    "grass":    (120, 200, 80),
    "ice":      (152, 216, 216),
    "fighting": (192, 48, 40),
    "poison":   (160, 64, 160),
    "ground":   (224, 192, 104),
    "flying":   (168, 144, 240),
    "psychic":  (248, 88, 136),
    "bug":      (168, 184, 32),
    "rock":     (184, 160, 56),
    "ghost":    (112, 88, 152),
    "dragon":   (112, 56, 248),
    "dark":     (112, 88, 72),
    "steel":    (184, 184, 208),
    "fairy":    (238, 153, 172),
}

_KNOWN_STATUS = {"par", "slp", "frz", "brn", "psn", "tox", "clean", "nostatus"}
_KNOWN_EFFECT = {"noeffect"}


# ── helpers ──────────────────────────────────────────────────────────

def _rgb(term, r: int, g: int, b: int) -> str:
    """Return term.color_rgb(r,g,b) or empty string if term is None."""
    if term is None:
        return ""
    return term.color_rgb(r, g, b)


def _on_rgb(term, r: int, g: int, b: int) -> str:
    """Return term.on_color_rgb(r,g,b) or empty string if term is None."""
    if term is None:
        return ""
    return term.on_color_rgb(r, g, b)


def turn_background(term, turn_index: int) -> str:
    """Return an ANSI background escape for the given 0‑based turn index."""
    r, g, b = TURN_BG_COLORS[turn_index % len(TURN_BG_COLORS)]
    return _on_rgb(term, r, g, b)


# ── state‑block syntax highlighting ──────────────────────────────────

def colorize_state_line(text: str, term=None) -> str:
    """Wrap tokens in a detokenized state line with ANSI colour codes."""
    if term is None:
        return text

    nc = term.normal
    dim = term.dim
    bold = term.bold

    # ── structural tag colours ──
    tag_color: dict[str, str] = {}

    green = _rgb(term, 120, 220, 100)
    red = _rgb(term, 240, 100, 100)
    yellow = _rgb(term, 240, 220, 80)
    blue = _rgb(term, 100, 160, 240)
    magenta = _rgb(term, 210, 130, 240)
    cyan = _rgb(term, 80, 200, 200)

    # Structural / arena / conditions / boosts — dim cyan
    for t in ("<bos>", "<eos>", "<boa>", "<eoa>",
              "<format>", "<end_format>", "<turn>", "<end_turn>",
              "<arena>", "<end_arena>", "<conditions>", "<end_conditions>",
              "<empty_conditions>", "<conditions_empty>", "<boosts>", "<end_boosts>"):
        tag_color[t] = dim + cyan
    # Active Pokémon
    for t in ("<active>", "<end_active>"):
        tag_color[t] = green
    # Opponent Pokémon
    for t in ("<opponent>", "<end_opponent>"):
        tag_color[t] = red
    # Moves section
    for t in ("<begin_moves>", "<end_moves>", "<move>", "<end_move>"):
        tag_color[t] = yellow
    # Bench section
    for t in ("<bench>", "<end_bench>"):
        tag_color[t] = blue
    # Species tags
    for t in ("<poke1>", "<end_poke1>", "<poke2>", "<end_poke2>",
              "<poke3>", "<end_poke3>", "<poke4>", "<end_poke4>",
              "<poke5>", "<end_poke5>", "<poke6>", "<end_poke6>"):
        tag_color[t] = magenta
    # Action delimiters
    for t in ("<chosen_move>", "<end_chosen_move>",
              "<opponent_chosen_move>", "<end_opponent_chosen_move>"):
        tag_color[t] = dim + yellow

    parts: list[str] = []
    for token in text.split():
        if token in tag_color:
            parts.append(tag_color[token] + token + nc)
        elif token in _TYPE_COLORS:
            r, g, b = _TYPE_COLORS[token]
            parts.append(_rgb(term, r, g, b) + token + nc)
        elif token in _KNOWN_STATUS:
            parts.append(dim + red + token + nc)
        elif token in _KNOWN_EFFECT:
            parts.append(dim + cyan + token + nc)
        elif token.startswith("spa") or token.startswith("atk") or \
             token.startswith("spd") or token.startswith("spe") or \
             token.startswith("def"):
            parts.append(magenta + token + nc)
        elif token.replace(".", "", 1).isdigit():
            parts.append(bold + token + nc)
        else:
            parts.append(token)
    return " ".join(parts)


# ── raw protocol line colouring ──────────────────────────────────────

_RAW_LINE_COLORS: dict[str, str] = {}  # filled once term is available


def _ensure_raw_line_colors(term) -> None:
    """Build (or rebuild) the raw‑line colour map from *term*."""
    if _RAW_LINE_COLORS or term is None:
        return
    dim = term.dim
    bold = term.bold
    green = _rgb(term, 120, 220, 100)
    red = _rgb(term, 240, 100, 100)
    yellow = _rgb(term, 240, 220, 80)
    blue = _rgb(term, 100, 160, 240)
    magenta = _rgb(term, 210, 130, 240)
    cyan = _rgb(term, 80, 200, 200)
    orange = _rgb(term, 240, 180, 60)
    white = _rgb(term, 255, 255, 255)
    grey = _rgb(term, 140, 140, 140)
    dark_grey = _rgb(term, 80, 80, 80)

    raw: dict[str, str] = {
        "init": dim + cyan,
        "title": white + bold,
        "j": dim + yellow,
        "player": green,
        "gen": dim + cyan,
        "tier": dim + cyan,
        "gametype": dim + cyan,
        "rule": dim + magenta,
        "teamsize": dim,
        "start": green + bold,
        "switch": green,
        "drag": green,
        "move": yellow + bold,
        "-damage": red,
        "-status": magenta,
        "-curestatus": green,
        "-boost": blue,
        "-unboost": blue,
        "-ability": cyan + bold,
        "-endability": cyan,
        "-item": orange,
        "-enditem": orange,
        "-sidestart": blue,
        "-sideend": blue,
        "-weather": cyan + bold,
        "-fieldstart": green + bold,
        "-fieldend": green,
        "-start": dim + blue,
        "-end": dim + blue,
        "-activate": magenta,
        "-transform": magenta + bold,
        "faint": red + bold,
        "turn": white + bold,
        "upkeep": dim + grey,
        "-fail": red,
        "-crit": yellow + bold,
        "-supereffective": yellow,
        "-sethp": red,
        "-hint": dim + grey,
        "win": green + bold,
        "tie": white + bold,
        "request": dim + grey,
        "choice": dim + yellow,
        "replace": magenta + bold,
        "cant": red,
        "chat": dim + grey,
        "c": dim + grey,
        "-message": dim + grey,
        "raw": dim + grey,
        "-nothing": dim,
        "t:": dark_grey,
        "": dim,
    }
    _RAW_LINE_COLORS.update(raw)


def colorize_raw_line(line: str, term=None, turn_bg: str = "") -> str:
    """Apply ANSI colour to a raw protocol line based on its message type."""
    if term is None:
        return turn_bg + line if turn_bg else line

    _ensure_raw_line_colors(term)
    nc = term.normal

    first_pipe = line.split("|")[1] if "|" in line else ""
    color = _RAW_LINE_COLORS.get(first_pipe, term.dim)
    return f"{turn_bg}{color}{line}{nc}"
