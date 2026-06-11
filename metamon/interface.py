from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
)
from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym
import string

from poke_env.environment import (
    Battle,
    Move,
    Pokemon,
    Field,
    Effect,
    SideCondition,
    Status,
    PokemonType,
)
from poke_env.player import BattleOrder, Player

import metamon
from metamon.config import format_for_agent
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.backend.replay_parser.replay_state import (
    Move as ReplayMove,
    Pokemon as ReplayPokemon,
    Action as ReplayAction,
    ReplayState,
    Nothing as ReplayNothing,
)
from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    clean_name,
    pokemon_name,
    move_name,
)

# Global registries mapping string names to component constructors.
# Populated by the @register_* decorators; consumed by get_observation_space(),
# get_action_space(), and get_reward_function() for string-based instantiation
# from configs or CLI arguments.
ALL_OBSERVATION_SPACES: Dict[str, Type["ObservationSpace"]] = {}
ALL_ACTION_SPACES: Dict[str, Type["ActionSpace"]] = {}
ALL_REWARD_FUNCTIONS: Dict[str, Type["RewardFunction"]] = {}


def register_observation_space(name: Optional[str] = None) -> Callable[[Type["ObservationSpace"]], Type["ObservationSpace"]]:
    """Decorator that registers an ObservationSpace subclass under a string name.

    Registered classes are callable via ``get_observation_space(name)``, enabling
    config-driven instantiation (e.g. CLI ``--obs_space DefaultObservationSpace``).

    Args:
        name: Key to register under. Defaults to ``cls.__name__`` if not given.

    Raises:
        ValueError: If the name is already taken in the global registry.
    """

    def _register(cls: Type["ObservationSpace"]) -> Type["ObservationSpace"]:
        obs_name = name if name is not None else cls.__name__
        if obs_name in ALL_OBSERVATION_SPACES:
            raise ValueError(f"Observation space '{obs_name}' is already registered!")
        ALL_OBSERVATION_SPACES[obs_name] = cls
        return cls

    return _register


def register_action_space(name: Optional[str] = None) -> Callable[[Type["ActionSpace"]], Type["ActionSpace"]]:
    """Decorator that registers an ActionSpace subclass under a string name.

    Mirror of ``register_observation_space`` for action spaces. Enables
    ``get_action_space(name)`` for config-driven selection of how agent outputs
    are mapped to ``UniversalAction`` indices.
    """

    def _register(cls: Type["ActionSpace"]) -> Type["ActionSpace"]:
        action_name = name if name is not None else cls.__name__
        if action_name in ALL_ACTION_SPACES:
            raise ValueError(f"Action space '{action_name}' is already registered!")
        ALL_ACTION_SPACES[action_name] = cls
        return cls

    return _register


def register_reward_function(name: Optional[str] = None) -> Callable[[Type["RewardFunction"]], Type["RewardFunction"]]:
    """Decorator that registers a RewardFunction subclass under a string name.

    Mirror of ``register_observation_space`` for reward functions. Enables
    ``get_reward_function(name)``.
    """

    def _register(cls: Type["RewardFunction"]) -> Type["RewardFunction"]:
        reward_name = name if name is not None else cls.__name__
        if reward_name in ALL_REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{reward_name}' is already registered!")
        ALL_REWARD_FUNCTIONS[reward_name] = cls
        return cls

    return _register


def get_observation_space_names() -> List[str]:
    """Return sorted list of all registered observation space names."""
    return sorted(ALL_OBSERVATION_SPACES.keys())


def get_action_space_names() -> List[str]:
    """Return sorted list of all registered action space names."""
    return sorted(ALL_ACTION_SPACES.keys())


def get_reward_function_names() -> List[str]:
    """Return sorted list of all registered reward function names."""
    return sorted(ALL_REWARD_FUNCTIONS.keys())


def get_observation_space(name: str) -> "ObservationSpace":
    """Look up and instantiate a registered observation space by name.

    Calls the class constructor with no arguments, so the class must accept
    ``__init__(self)`` or use defaults.
    """
    if name not in ALL_OBSERVATION_SPACES:
        raise ValueError(
            f"Unknown observation space '{name}' (available: {get_observation_space_names()})"
        )
    return ALL_OBSERVATION_SPACES[name]()


def get_action_space(name: str) -> "ActionSpace":
    """Look up and instantiate a registered action space by name."""
    if name not in ALL_ACTION_SPACES:
        raise ValueError(
            f"Unknown action space '{name}' (available: {get_action_space_names()})"
        )
    return ALL_ACTION_SPACES[name]()


def get_reward_function(name: str) -> "RewardFunction":
    """Look up and instantiate a registered reward function by name."""
    if name not in ALL_REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}' (available: {get_reward_function_names()})"
        )
    return ALL_REWARD_FUNCTIONS[name]()


def consistent_pokemon_order(pokemon: List[Any]) -> List[Any]:
    """Sort a list of Pokémon alphabetically by cleaned species name.

    This deterministic ordering is critical: action indices for switches (4–8) depend
    on the sort order of the available benchmark. Without a consistent order, the same
    switch target could map to different indices across backends or parses.

    Accepts lists of ``poke_env.Pokemon``, ``ReplayPokemon``, ``UniversalPokemon``,
    or plain strings.  All are keyed by their species name after passing through
    ``pokemon_name()`` (lowercase, special characters stripped).
    """
    if not pokemon:
        return []
    if isinstance(pokemon[0], Pokemon):
        key = lambda p: pokemon_name(p.species)
    elif isinstance(pokemon[0], str):
        key = lambda p: pokemon_name(p)
    elif isinstance(pokemon[0], UniversalPokemon):
        key = lambda p: pokemon_name(p.name)
    elif isinstance(pokemon[0], ReplayPokemon):
        key = lambda p: pokemon_name(p.name)
    else:
        raise ValueError(
            f"Unrecognized `pokemon` list format of type {type(pokemon)}: {pokemon}"
        )
    return sorted(pokemon, key=key)


def consistent_move_order(moves: List[Any]) -> List[Any]:
    """Sort a list of moves alphabetically by cleaned move name.

    The same determinism requirement as ``consistent_pokemon_order`` applies:
    action indices for moves (0–3, 9–12) depend on this sort order, so it must
    be stable and backend-agnostic.

    Accepts lists of ``poke_env.Move``, ``ReplayMove``, ``UniversalMove``,
    or plain strings.
    """
    if not moves:
        return []
    if isinstance(moves[0], Move):
        key = lambda m: move_name(m.id)
    elif isinstance(moves[0], str):
        key = lambda m: move_name(m)
    elif isinstance(moves[0], UniversalMove):
        key = lambda m: move_name(m.name)
    elif isinstance(moves[0], ReplayMove):
        key = lambda m: move_name(m.name)
    else:
        raise ValueError(
            f"Unrecognized `moves` list format of type {type(moves[0])}: {moves}"
        )
    return sorted(moves, key=key)


# Module-level cache for UniversalMove.from_Move (static move properties never change)
_UNIVERSAL_MOVE_CACHE: dict[str, "UniversalMove"] = {}


@dataclass
class UniversalMove:
    """A move represented in the backend-agnostic Universal format.

    Provides static move metadata (name, type, category, base power, accuracy,
    priority) plus dynamic per-battle state (current PP, max PP). This class is
    the common currency between the three data sources (poke-env online battles,
    replay-parser offline trajectories, and on-disk parsed JSON datasets).

    Rarely constructed directly — use the factory classmethods:

        * ``from_Move()`` — from a poke-env ``Move``
        * ``from_ReplayMove()`` — from a replay-parser ``Move``
        * ``from_dict()`` — from a JSON dict (dataset on disk)
    """

    name: str
    move_type: str
    category: str
    base_power: int
    accuracy: float
    priority: int
    current_pp: int
    max_pp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "move_type": self.move_type,
            "category": self.category,
            "base_power": self.base_power,
            "accuracy": self.accuracy,
            "priority": self.priority,
            "current_pp": self.current_pp,
            "max_pp": self.max_pp,
        }

    @classmethod
    def blank_move(cls) -> "UniversalMove":
        """Return a sentinel UniversalMove representing 'no move'.

        Used when a Pokémon has no previous move (e.g. first turn of the battle)
        or the move is unknown.  All fields use obvious placeholder strings/zeros.
        """
        return cls(
            name="<blank>",
            move_type="notype",
            category="status",
            base_power=0,
            accuracy=0.0,
            priority=0,
            current_pp=0,
            max_pp=0,
        )

    @classmethod
    def from_ReplayMove(cls, move: Optional[ReplayMove]) -> "UniversalMove":
        """Build from a replay-parser ``Move``, including dynamic PP state.

        Delegates to ``from_Move`` for static properties, then layers on the
        replay-parser-specific PP values.  If ``move`` is None, returns a blank.
        """
        universal_move = cls.from_Move(move)
        if move is not None:
            universal_move.current_pp = move.pp
            universal_move.max_pp = move.maximum_pp
        return universal_move

    @classmethod
    def from_Move(cls, move: Optional[Move]) -> "UniversalMove":
        """Build from a poke-env ``Move``, caching static properties by move ID.

        Static properties (name, type, category, base_power, accuracy, priority)
        never change for the same move ID, so they are memoized in a module-level
        cache to avoid repeated string cleaning and object allocation.
        Dynamic PP is *not* cached and is read fresh from the move object.
        """
        if move is None:
            return cls.blank_move()
        assert isinstance(move, Move)
        # Cache by move ID — static properties never change for the same move
        mid = move.id
        cached = _UNIVERSAL_MOVE_CACHE.get(mid)
        if cached is not None:
            um = UniversalMove(
                name=cached.name,
                category=cached.category,
                base_power=cached.base_power,
                move_type=cached.move_type,
                priority=cached.priority,
                accuracy=cached.accuracy,
                current_pp=move.current_pp,
                max_pp=move.max_pp,
            )
            return um
        um = cls(
            name=move_name(move.id),
            category=clean_name(move.category.name),
            base_power=move.base_power,
            move_type=clean_name(move.type.name),
            priority=move.priority,
            accuracy=move.accuracy,
            current_pp=move.current_pp,
            max_pp=move.max_pp,
        )
        _UNIVERSAL_MOVE_CACHE[mid] = um
        return um


@dataclass
class UniversalPokemon:
    """A Pokémon represented in the backend-agnostic Universal format.

    Holds both static species data (base stats, types, Tera type, base species)
    and dynamic battle state (HP %, item, ability, status, effects, boosts,
    current moves with PP).  This is the single Pokémon representation used by
    all observation spaces, datasets, and the RL environment.

    Rarely constructed directly — use the factory classmethods:

        * ``from_ReplayPokemon()`` — from the replay parser's ``Pokemon``
        * ``from_Pokemon()`` — from poke-env's ``Pokemon``
        * ``from_dict()`` — from a JSON dict (dataset on disk)

    .. note::

        Movesets may exceed 4 entries due to Transform / Mimic edge cases in
        the replay parser.  All factory methods truncate to 4 moves via ``[:4]``
        as a safety net.
    """

    name: str
    hp_pct: float
    types: str
    item: str
    ability: str
    lvl: int
    status: str
    effect: str
    moves: list[UniversalMove]

    atk_boost: int
    spa_boost: int
    def_boost: int
    spd_boost: int
    spe_boost: int
    accuracy_boost: int
    evasion_boost: int

    base_atk: int
    base_spa: int
    base_def: int
    base_spd: int
    base_spe: int
    base_hp: int

    # version-specific
    tera_type: str
    base_species: str

    @staticmethod
    def universal_items(item_rep: Optional[str | ReplayNothing]) -> str:
        """Normalize an item representation to a canonical string token.

        Handles the multiple ways 'no item' can appear across backends:
        ``None``, ``"unknown_item"``, empty string, ``"No Item"``,
        ``"noitem"``, and the ``ReplayNothing.NO_ITEM`` sentinel.
        All are mapped to ``"noitem"`` (after ``clean_no_numbers``).
        Unknown items become ``"unknownitem"``.
        """
        if item_rep is None or item_rep == "unknown_item":
            item_str = "unknownitem"
        elif item_rep == ReplayNothing.NO_ITEM:
            item_str = item_rep.name
        elif isinstance(item_rep, str) and item_rep.strip() in {
            "",
            "No Item",
            "noitem",
        }:
            item_str = ReplayNothing.NO_ITEM.name
        else:
            item_str = item_rep
        return clean_no_numbers(item_str)

    @staticmethod
    def universal_abilities(ability_rep: Optional[str | ReplayNothing]) -> str:
        """Normalize an ability representation to a canonical string token.

        Same pattern as ``universal_items``: collapses None, unknown markers,
        empty strings, and ``ReplayNothing.NO_ABILITY`` into ``"noability"``
        or ``"unknownability"``.
        """
        if ability_rep is None or ability_rep == "unknown_ability":
            ability_str = "unknownability"
        elif ability_rep == ReplayNothing.NO_ABILITY:
            ability_str = ability_rep.name
        elif isinstance(ability_rep, str) and ability_rep.strip() in {
            "",
            "No Ability",
            "noability",
        }:
            ability_str = ReplayNothing.NO_ABILITY.name
        else:
            ability_str = ability_rep
        return clean_no_numbers(ability_str)

    @staticmethod
    def universal_effects(effect: Optional[Effect]) -> str:
        """Map a poke-env ``Effect`` to its cleaned name, or ``"noeffect"``."""
        if not effect:
            return "noeffect"
        return clean_no_numbers(effect.name)

    @staticmethod
    def universal_status(status_rep: Status | ReplayNothing) -> str:
        """Map a poke-env ``Status`` or ``ReplayNothing.NO_STATUS`` to a string.

        Missing/unknown status becomes ``"nostatus"``.
        """
        if status_rep is None or status_rep == ReplayNothing.NO_STATUS:
            return "nostatus"
        assert isinstance(status_rep, Status)
        return clean_no_numbers(status_rep.name)

    @staticmethod
    def universal_types(type_rep: list, force_two: bool = True) -> str:
        """Convert a list of types to a space-separated, sorted string.

        ``None`` entries or ``ReplayNothing.NO_TERA_TYPE`` become ``"notype"``.
        When ``force_two=True`` (the default for dual-type Pokémon), the list is
        padded to exactly two entries.  When ``force_two=False`` (used for Tera
        type, which is always a single type), the list is left as-is.

        Types are sorted alphabetically so dual-type order doesn't matter.
        """
        if force_two:
            while len(type_rep) < 2:
                type_rep.append(None)
        type_strs = []
        for type in type_rep:
            if type is None or type == ReplayNothing.NO_TERA_TYPE:
                type_strs.append("notype")
            elif isinstance(type, PokemonType):
                type_strs.append(clean_name(type.name))
            elif isinstance(type, str):
                type_strs.append(clean_name(type))
        return " ".join(sorted(type_strs))

    @classmethod
    def from_ReplayPokemon(cls, pokemon: ReplayPokemon) -> "UniversalPokemon":
        """Build from the replay parser's ``Pokemon`` object.

        Extracts base stats, stat boosts, current HP fraction, active item/ability,
        and the most recently-applied volatile effect.  Moves are truncated to 4.

        Guards against ``None`` HP (unrevealed opponent Pokémon during team preview)
        by defaulting to 100/100.
        """
        moves = [
            UniversalMove.from_ReplayMove(move)
            for move in pokemon.moves.values()
            if move is not None
        ][:4]
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        boosts = {
            f"{stat}boost": getattr(pokemon.boosts, stat)
            for stat in pokemon.boosts.stat_attrs
        }
        if pokemon.effects:
            most_recent_effect = min(pokemon.effects.keys(), key=pokemon.effects.get)
        else:
            most_recent_effect = None
        # Guard against None HP (unrevealed opponent Pokémon from team preview)
        cur_hp = pokemon.current_hp if pokemon.current_hp is not None else 100
        max_hp = pokemon.max_hp if pokemon.max_hp is not None else 100
        return cls(
            name=pokemon_name(pokemon.name),
            base_species=pokemon_name(pokemon.had_name),
            hp_pct=round(float(cur_hp) / max_hp, 2),
            types=cls.universal_types(pokemon.type),
            tera_type=cls.universal_types([pokemon.tera_type], force_two=False),
            item=cls.universal_items(pokemon.active_item),
            ability=cls.universal_abilities(pokemon.active_ability),
            lvl=pokemon.lvl,
            status=cls.universal_status(pokemon.status),
            effect=cls.universal_effects(most_recent_effect),
            moves=moves,
            **(boosts | stats),
        )

    @classmethod
    def from_Pokemon(cls, pokemon: Pokemon) -> "UniversalPokemon":
        """Build from a poke-env ``Pokemon`` object.

        Same extraction logic as ``from_ReplayPokemon`` but uses poke-env's
        attribute names (``.species``, ``.current_hp_fraction``, ``.boosts``
        dict, etc.).  Moves are truncated to 4.

        .. warning::
            Do NOT use ``Battle.available_moves`` when building a UniversalPokemon
            for an observation — it may exclude disabled moves the agent still
            needs to see.  Use ``pokemon.moves`` directly.
        """
        moves = [UniversalMove.from_Move(move) for move in pokemon.moves.values()][:4]
        boosts = {f"{stat}_boost": boost for stat, boost in pokemon.boosts.items()}
        stats = {f"base_{stat}": val for stat, val in pokemon.base_stats.items()}
        if pokemon.effects:
            most_recent_effect = min(pokemon.effects.keys(), key=pokemon.effects.get)
        else:
            most_recent_effect = None
        return cls(
            name=pokemon_name(pokemon.species),
            base_species=pokemon_name(pokemon.base_species),
            hp_pct=round(float(pokemon.current_hp_fraction), 2),
            types=cls.universal_types(pokemon.types),
            tera_type=cls.universal_types([pokemon.tera_type], force_two=False),
            item=cls.universal_items(pokemon.item),
            ability=cls.universal_abilities(pokemon.ability),
            lvl=pokemon.level,
            status=cls.universal_status(pokemon.status),
            effect=cls.universal_effects(most_recent_effect),
            moves=moves,
            **(boosts | stats),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for on-disk storage.

        Nested moves are serialized via ``UniversalMove.to_dict()``.
        """
        return {
            "name": self.name,
            "hp_pct": self.hp_pct,
            "types": self.types,
            "item": self.item,
            "ability": self.ability,
            "lvl": self.lvl,
            "status": self.status,
            "effect": self.effect,
            "moves": [m.to_dict() for m in self.moves],
            "atk_boost": self.atk_boost,
            "spa_boost": self.spa_boost,
            "def_boost": self.def_boost,
            "spd_boost": self.spd_boost,
            "spe_boost": self.spe_boost,
            "accuracy_boost": self.accuracy_boost,
            "evasion_boost": self.evasion_boost,
            "base_atk": self.base_atk,
            "base_spa": self.base_spa,
            "base_def": self.base_def,
            "base_spd": self.base_spd,
            "base_spe": self.base_spe,
            "base_hp": self.base_hp,
            "tera_type": self.tera_type,
            "base_species": self.base_species,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalPokemon":
        """Deserialize from a JSON dict (offline dataset on disk).

        Handles backwards compatibility: missing ``tera_type`` defaults to
        ``"notype"`` (pre-gen-9 datasets), missing ``base_species`` is inferred
        from the display name by stripping forme suffixes (e.g. ``"Rotom-Wash"``
        → ``"Rotom"``).  Moves are truncated to 4 as a safety net against
        Transform/Mimic edge cases that can produce >4 move entries.
        """
        data["moves"] = [UniversalMove(**m) for m in data["moves"][:4]]
        if "tera_type" not in data:
            data["tera_type"] = cls.universal_types([None], force_two=False)
        if "base_species" not in data:
            data["base_species"] = data["name"].split("-")[0].strip()
        return cls(**data)

    @staticmethod
    def metamon_to_poke_env(pokemon: Optional[ReplayPokemon], is_active: bool) -> Optional[Pokemon]:
        """
        Straight-through conversion from metamon replay parser Pokemon object
        to poke-env Pokemon object. An ugly alternative to adding a
        `update_from_metamon` equivalent in poke-env.Pokemon. Used by metamon
        battle backend.
        """
        if pokemon is None:
            return None
        p = Pokemon(gen=pokemon.gen)
        p._base_stats = pokemon.base_stats
        p._type_1 = PokemonType.from_name(pokemon.type[0])
        p._type_2 = (
            PokemonType.from_name(pokemon.type[1]) if len(pokemon.type) > 1 else None
        )
        p._ability = pokemon.had_ability
        p._level = pokemon.lvl
        p._max_hp = pokemon.max_hp
        p._moves = {m.lookup_name: m for m in pokemon.moves.values()}
        for m in p._moves.values():
            m.set_pp(m.pp)
        p._name = pokemon.nickname
        p._species = clean_name(pokemon.name)
        p._active = is_active
        p._boosts = pokemon.boosts.to_dict()
        p._current_hp = pokemon.current_hp
        p._effects = pokemon.effects
        p._item = pokemon.active_item
        p._status = pokemon.status
        p._temporary_ability = pokemon.active_ability
        p._previous_move = pokemon.last_used_move
        p._terastallized_type = (
            PokemonType.from_name(pokemon.tera_type) if pokemon.tera_type else None
        )
        return p


@dataclass
class UniversalState:
    """A battle state represented in the backend-agnostic Universal format.

    This is the single state representation consumed by all observation spaces,
    reward functions, and datasets.  It captures everything visible from one
    player's point of view at a single timestep.

    Fields are split into several logical groups:

    * **Active Pokémon** — the player's and opponent's current Pokémon with
      full HP, stats, boosts, status, moves, etc.
    * **Bench** — switchable teammates (``available_switches``), revealed
      opponent bench, and fainted Pokémon (own + opponent).
    * **Battle context** — format, weather, side conditions, battle field,
      forced switch flag.
    * **History** — previous moves used by both active Pokémon.
    * **Terminal** — ``battle_won`` / ``battle_lost`` flags.
    * **Version-specific** — ``can_tera`` (gen 9 only), ``opponent_teampreview``.

    Rarely constructed directly — use the factory classmethods:

        * ``from_ReplayState()`` — from the replay parser's ``ReplayState``
        * ``from_Battle()`` — from poke-env's ``Battle``
        * ``from_dict()`` — from a JSON dict (dataset on disk)
    """

    format: str
    player_active_pokemon: UniversalPokemon
    opponent_active_pokemon: UniversalPokemon
    available_switches: List[UniversalPokemon]
    opponent_bench: List[UniversalPokemon]  # opponent's non-active, non-fainted, revealed pokemon
    fainted_pokemon: List[UniversalPokemon]  # player's fainted pokemon (may be revived in gen9)
    opponent_fainted: List[UniversalPokemon]  # opponent's fainted, revealed pokemon
    player_prev_move: UniversalMove
    opponent_prev_move: UniversalMove
    opponents_remaining: int
    player_conditions: str
    opponent_conditions: str
    weather: str
    battle_field: str
    forced_switch: bool
    battle_won: bool
    battle_lost: bool

    # version-specific
    can_tera: bool  # added v3-beta
    opponent_teampreview: List[str]  # added v3

    @property
    def agent_format(self) -> str:
        """The format as presented to the agent, with Showdown variants normalized."""
        return format_for_agent(self.format)

    @staticmethod
    def universal_conditions(condition_rep: Any) -> str:
        if not condition_rep:
            return "noconditions"
        most_recent = max(condition_rep.keys(), key=condition_rep.get)
        assert isinstance(most_recent, SideCondition)
        return clean_no_numbers(most_recent.name)

    @staticmethod
    def universal_field(field_rep: Any) -> str:
        if not field_rep:
            return "nofield"
        most_recent = max(field_rep.keys(), key=field_rep.get)
        assert isinstance(most_recent, Field)
        return clean_no_numbers(most_recent.name)

    @staticmethod
    def universal_weather(weather_rep: Any) -> str:
        if not weather_rep or weather_rep == ReplayNothing.NO_WEATHER:
            return "noweather"
        if isinstance(weather_rep, dict):
            weather_rep = list(weather_rep.keys())[0]
        return clean_no_numbers(weather_rep.name)

    # fmt: off
    @classmethod
    def from_ReplayState(cls, state: ReplayState) -> "UniversalState":
        assert isinstance(state, ReplayState)
        format = re.sub(r"\[|\]| ", "", state.format).lower()
        active = UniversalPokemon.from_ReplayPokemon(state.active_pokemon)
        opponent = UniversalPokemon.from_ReplayPokemon(state.opponent_active_pokemon)
        switches = [UniversalPokemon.from_ReplayPokemon(p) for p in consistent_pokemon_order(state.available_switches)]
        # opponent bench: non-active, non-fainted, revealed pokemon from opponent_team
        active_opp_id = state.opponent_active_pokemon.unique_id if state.opponent_active_pokemon else None
        opponent_bench = [
            UniversalPokemon.from_ReplayPokemon(p)
            for p in state.opponent_team
            if p is not None
            and p.status != Status.FNT
            and p.unique_id != active_opp_id
        ]
        # player fainted: own fainted pokemon (non-active)
        active_self_id = state.active_pokemon.unique_id if state.active_pokemon else None
        fainted_pokemon = [
            UniversalPokemon.from_ReplayPokemon(p)
            for p in (state.player_team if state.player_team else [])
            if p is not None
            and p.status == Status.FNT
            and p.unique_id != active_self_id
        ]
        # opponent fainted: revealed, fainted opponent pokemon (non-active)
        opponent_fainted = [
            UniversalPokemon.from_ReplayPokemon(p)
            for p in state.opponent_team
            if p is not None
            and p.status == Status.FNT
            and p.unique_id != active_opp_id
        ]
        opponents_remaining = 6 - sum(p.status == Status.FNT for p in state.opponent_team if p is not None)
        opponent_teampreview = [pokemon_name(p.had_name) for p in state.opponent_teampreview]
        return cls(
            format=format,
            player_active_pokemon=active,
            opponent_active_pokemon=opponent,
            available_switches=switches,
            opponent_bench=opponent_bench,
            fainted_pokemon=fainted_pokemon,
            opponent_fainted=opponent_fainted,
            player_prev_move=UniversalMove.from_ReplayMove(state.player_prev_move),
            opponent_prev_move=UniversalMove.from_ReplayMove(state.opponent_prev_move),
            player_conditions=cls.universal_conditions(state.player_conditions),
            opponent_conditions=cls.universal_conditions(state.opponent_conditions),
            weather=cls.universal_weather(state.weather),
            battle_field=cls.universal_field(state.battle_field),
            forced_switch=state.force_switch,
            opponents_remaining=opponents_remaining,
            battle_won=state.battle_won,
            battle_lost=state.battle_lost,
            can_tera=state.can_tera,
            opponent_teampreview=opponent_teampreview,
        )

    @classmethod
    def from_Battle(cls, battle: Battle) -> "UniversalState":
        # do not use Battle.available_switches or Battle.available_moves
        format = battle.battle_tag.split("-")[1]
        weather = cls.universal_weather(battle.weather)
        battle_field = cls.universal_field(battle.fields)
        player_conditions = cls.universal_conditions(battle.side_conditions)
        opponent_conditions = cls.universal_conditions(battle.opponent_side_conditions)
        active = UniversalPokemon.from_Pokemon(battle.active_pokemon)
        opponent = UniversalPokemon.from_Pokemon(battle.opponent_active_pokemon)
        if battle.reviving:
            possible_switches = [p for p in battle.team.values() if p.fainted and not p.active]
        else:
            possible_switches = [p for p in battle.team.values() if not p.fainted and not p.active]
        switches = [UniversalPokemon.from_Pokemon(p) for p in consistent_pokemon_order(possible_switches)]
        player_prev_move = UniversalMove.from_Move(battle.active_pokemon.previous_move)
        opponent_prev_move = UniversalMove.from_Move(battle.opponent_active_pokemon.previous_move)
        # NOTE: always assumes 6 in the party, and this will probably never change for backwards compat
        opponents_remaining = 6 - sum(p.status == Status.FNT for p in battle.opponent_team.values())
        opponent_teampreview = [pokemon_name(p.base_species) for p in battle.teampreview_opponent_team if p is not None]
        force_switch = battle.force_switch
        if isinstance(force_switch, list):
            force_switch = force_switch[0]

        return cls(
            format=format,
            player_active_pokemon=active,
            opponent_active_pokemon=opponent,
            available_switches=switches,
            opponent_bench=[],  # poke-env backend does not provide opponent bench
            fainted_pokemon=[],  # poke-env backend does not provide fainted tracking
            opponent_fainted=[],
            player_prev_move=player_prev_move,
            opponent_prev_move=opponent_prev_move,
            player_conditions=player_conditions,
            opponent_conditions=opponent_conditions,
            weather=weather,
            battle_field=battle_field,
            forced_switch=force_switch,
            battle_won=battle.won if battle.won else False,
            battle_lost=battle.lost if battle.lost else False,
            opponents_remaining=opponents_remaining,
            can_tera=battle.can_tera is not None,
            opponent_teampreview=opponent_teampreview,
        )
    # fmt: on

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for on-disk storage.

        All nested ``UniversalPokemon`` and ``UniversalMove`` objects are
        recursively serialized via their own ``to_dict()`` methods.
        """
        return {
            "format": self.format,
            "player_active_pokemon": self.player_active_pokemon.to_dict(),
            "opponent_active_pokemon": self.opponent_active_pokemon.to_dict(),
            "available_switches": [p.to_dict() for p in self.available_switches],
            "opponent_bench": [p.to_dict() for p in self.opponent_bench],
            "fainted_pokemon": [p.to_dict() for p in self.fainted_pokemon],
            "opponent_fainted": [p.to_dict() for p in self.opponent_fainted],
            "player_prev_move": self.player_prev_move.to_dict(),
            "opponent_prev_move": self.opponent_prev_move.to_dict(),
            "opponents_remaining": self.opponents_remaining,
            "player_conditions": self.player_conditions,
            "opponent_conditions": self.opponent_conditions,
            "weather": self.weather,
            "battle_field": self.battle_field,
            "forced_switch": self.forced_switch,
            "battle_won": self.battle_won,
            "battle_lost": self.battle_lost,
            "can_tera": self.can_tera,
            "opponent_teampreview": self.opponent_teampreview,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalState":
        # convert nested Pokemon objects
        data["player_active_pokemon"] = UniversalPokemon.from_dict(
            data["player_active_pokemon"]
        )
        data["opponent_active_pokemon"] = UniversalPokemon.from_dict(
            data["opponent_active_pokemon"]
        )
        data["available_switches"] = [
            UniversalPokemon.from_dict(p) for p in data["available_switches"]
        ]
        # convert nested Move objects
        data["player_prev_move"] = UniversalMove(**data["player_prev_move"])
        data["opponent_prev_move"] = UniversalMove(**data["opponent_prev_move"])

        if "can_tera" not in data:
            # backwards compat (if it's missing; it's an old version of the dataset
            # --> gen 1-4 --> no tera)
            data["can_tera"] = False

        if "opponent_teampreview" not in data:
            # backwards compat (if it's missing; it's an old version of the dataset
            # --> gen 1-4 --> no teampreview)
            data["opponent_teampreview"] = []

        if "opponent_bench" not in data:
            # backwards compat: old parsed replays don't have opponent bench
            data["opponent_bench"] = []
        else:
            data["opponent_bench"] = [
                UniversalPokemon.from_dict(p) for p in data["opponent_bench"]
            ]

        if "fainted_pokemon" not in data:
            data["fainted_pokemon"] = []
        else:
            data["fainted_pokemon"] = [
                UniversalPokemon.from_dict(p) for p in data["fainted_pokemon"]
            ]

        if "opponent_fainted" not in data:
            data["opponent_fainted"] = []
        else:
            data["opponent_fainted"] = [
                UniversalPokemon.from_dict(p) for p in data["opponent_fainted"]
            ]

        return cls(**data)


class UniversalAction:
    """A player action represented as an integer index in the Universal format.

    Maps the diverse action types (moves, switches, tera-moves, no-ops) into a
    single integer space, which is what models consume.  The index alone is
    sufficient to reconstruct the action given the corresponding ``UniversalState``
    (which provides the sorted move/switch lists the index indexes into).

    See ``from_ReplayAction()`` for the full index mapping table.
    """
    def __init__(self, action_idx: int) -> None:
        self.action_idx = action_idx

    @property
    def missing(self) -> bool:
        """``True`` if the action was never revealed (index -1).

        Missing actions occur when the player's choice is hidden because the
        Pokémon was paralyzed, asleep, flinched, or the choice was ambiguous
        due to Zoroark's Illusion.
        """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UniversalAction):
            return NotImplemented
        return self.action_idx == other.action_idx

    def __repr__(self) -> str:
        return str(self.action_idx)

    def __hash__(self) -> int:
        return hash(self.action_idx)

    @classmethod
    def from_ReplayAction(
        cls, state: ReplayState, action: ReplayAction
    ) -> Optional["UniversalAction"]:
        """Convert a spectator-perspective ``ReplayAction`` to a UniversalAction index.

        This is the canonical mapping from raw actions to the integer space:

        ============  ========================================================
        Index         Meaning
        ============  ========================================================
        -1            Missing / unrevealed action
        0             No-op: Recharge, Struggle, or Fight (Gen 1 attack button)
        1–3           Move (alphabetically sorted among active's 4 moves)
        4–8           Switch (alphabetically sorted among ≤5 available switches)
        9–12          Tera + move (index = 9 + move_index, Gen 9 only)
        ============  ========================================================

        Returns ``None`` if the action cannot be mapped (e.g. the move name
        doesn't match any known move in the active Pokémon's moveset).

        Edge cases handled:

        * Zoroark switch-to-self (disguised Zoroark appears to switch to the
          Pokémon it's impersonating) → treated as missing (index -1).
        * Tera animation shown but move never revealed → missing (index -1).
        """
        action_idx = None
        if action is None or (action.name is None and action.is_tera):
            # action was never revealed
            # (or tera animation was shown but the rest of the action was never revealed)
            action_idx = -1
        elif action.is_noop:
            assert action.name == "Recharge"
            action_idx = 0
        elif action.name in {"Struggle", "Fight"}:
            action_idx = 0
        elif action.is_switch or action.is_revival:
            # Switch-to-self can happen when Zoroark's Illusion is active
            # (the disguised Zoroark appears to switch to the Pokemon it's
            # impersonating).  Treat as a missing action.
            if action.target == state.active_pokemon:
                action_idx = -1
            else:
                for switch_idx, available_switch in enumerate(
                    consistent_pokemon_order(state.available_switches)
                ):
                    if available_switch.unique_id == action.target.unique_id:
                        action_idx = 4 + switch_idx
                        break
        else:
            move_options = list(state.active_pokemon.moves.values())
            for move_idx, move in enumerate(consistent_move_order(move_options)):
                if move.name == action.name:
                    action_idx = move_idx
                    if action.is_tera:
                        action_idx += 9
                    break
        if action_idx is None:
            return None
        return cls(action_idx)

    @classmethod
    def maybe_valid_actions(cls, state: UniversalState) -> Set["UniversalAction"]:
        """Return the set of *possibly* legal actions from a given state.

        "Possibly" is the key word: this method uses only the information
        available in a ``UniversalState`` (i.e., the offline dataset view).
        It does NOT have access to poke-env's ``Battle.available_moves``
        (which can disable moves due to Taunt, Disable, etc.).  Therefore
        some actions in the returned set may actually be illegal.

        The masking logic:

        * If ``forced_switch`` is True, only switch indices (4–8) are included.
        * Otherwise, move indices (0–3), tera move indices (9–12 if ``can_tera``),
          AND switch indices (4–8) are all included.
        * Indices 0–3 are filtered to the actual number of moves the active
          Pokémon has (usually 4, but can be fewer).

        This is used for legal-action masking during training/inference from
        offline datasets.  For online play where the full battle state is
        available, use ``definitely_valid_actions()`` instead.
        """
        legal = []
        if not state.forced_switch: # add moves if not force switch
            moves = len(state.player_active_pokemon.moves)
            legal.extend(range(moves))
            if state.can_tera:
                legal.extend(range(9, 9 + moves))
        legal.extend(range(4, 4 + len(state.available_switches))) # adding switches which start from 4
        return set(UniversalAction(action_idx=action_idx) for action_idx in legal)

    @classmethod
    def definitely_valid_actions(
        cls, state: UniversalState, battle: Battle
    ) -> Set["UniversalAction"]:
        """Return the set of *definitely* legal actions using poke-env's full battle state.

        Unlike ``maybe_valid_actions``, this method cross-references each candidate
        action with ``action_idx_to_BattleOrder()``, which checks poke-env's
        ``Battle.available_moves`` and ``Battle.available_switches``.  Only actions
        that produce a non-None ``BattleOrder`` are included.

        Used for online RL environment action masking where the full request
        data is available.
        """
        maybe_legal = cls.maybe_valid_actions(state)
        definitely_legal = set()
        for action in maybe_legal:
            order = cls.action_idx_to_BattleOrder(battle, action_idx=action.action_idx)
            if order is not None:
                definitely_legal.add(action)
        return definitely_legal

    @staticmethod
    def action_idx_to_BattleOrder(
        battle: Battle, action_idx: int
    ) -> Optional[BattleOrder]:
        """Convert a Universal action index to a poke-env ``BattleOrder``.

        This is the bridge from the integer model output to the online environment.
        It handles several special cases:

        * **Recharge** — ``available_moves`` is ``{"recharge"}``; the single
          recharge option is always returned regardless of action_idx.
        * **Struggle** — all move indices 0–3 map to Struggle (the agent sees
          its 4 moves but all of them execute Struggle).
        * **Fight** (Gen 1) — same override as Struggle: all move indices map
          to the "Fight" button.
        * **Tera** — indices 9–12 are detected via ``action_idx >= 9`` and
          the Tera flag is set on the ``BattleOrder``.

        Returns ``None`` if the index selects an invalid move/switch.  The
        caller (environment) is responsible for handling invalid orders via
        ``on_invalid_order``.
        """
        valid_moves = {m.id for m in battle.available_moves}
        if valid_moves == {"recharge"}:
            # there is only one option; take it so it doesn't count as an invalid action
            return Player.create_order(battle.available_moves[0])
        elif valid_moves == {"struggle"}:
            # override the options so that all the move indices are struggle but switches are valid.
            # note that the replay version sets every Struggle in the dataset to index 0, so this
            # is giving a little room for error.
            move_options = [battle.available_moves[0]] * 4
        elif "fight" in valid_moves:
            # new in ~march 2026: a "fight" button in gen1, which tells you your only
            # options are to "fight" or potentially switch. Similar to struggle, the agent
            # will see its regular 4 moves and switches (if applicable) but all of the moves
            # will map to clicking "fight".
            move_options = [battle.available_moves[0]] * 4
        else:
            # standard: pick from the active pokemon's moves
            move_options = consistent_move_order(
                list(battle.active_pokemon.moves.values())
            )

        valid_switches = {p.name for p in battle.available_switches}
        if not battle.reviving:
            switch_options = consistent_pokemon_order(
                [
                    p
                    for p in list(battle.team.values())
                    if not p.fainted and not p.active
                ]
            )
        else:
            switch_options = consistent_pokemon_order(
                [p for p in list(battle.team.values()) if p.fainted and not p.active]
            )

        wants_tera = False
        can_tera = battle.can_tera is not None
        if action_idx >= 9:
            wants_tera = True
            action_idx -= 9

        if action_idx <= 3 and not battle.force_switch:
            # pick one of up to 4 available moves
            if action_idx < len(move_options):
                selected_move = move_options[action_idx]
                if selected_move.id in valid_moves:
                    # NOTE: giving the player a little help on invalid tera requests here
                    order = Player.create_order(
                        selected_move, terastallize=wants_tera and can_tera
                    )
                    return order
        if 4 <= action_idx <= 8:
            # switch to one of up to 5 alternative pokemon
            action_idx -= 4
            if action_idx < len(switch_options):
                selected_switch = switch_options[action_idx]
                if selected_switch.name in valid_switches:
                    order = Player.create_order(selected_switch)
                    return order

        # Q: "what happens when we pick an invalid action? (order = None)"
        # A : up to env's `on_invalid_order` to pick one
        return None

    def to_BattleOrder(self, battle: Battle) -> Optional[BattleOrder]:
        """Instance wrapper around the static ``action_idx_to_BattleOrder``."""
        return UniversalAction.action_idx_to_BattleOrder(
            battle, action_idx=self.action_idx
        )


class ActionSpace(ABC):
    """Abstract interface for mapping between model outputs and UniversalAction indices.

    Subclasses define the gym action space shape and translate raw agent outputs
    (integers, logits, etc.) to/from ``UniversalAction`` objects.  This abstraction
    allows different action parameterizations (e.g. with/without Tera) without
    changing the rest of the pipeline.
    """

    @property
    @abstractmethod
    def gym_space(self) -> gym.spaces.Space:
        raise NotImplementedError

    @abstractmethod
    def agent_output_to_action(
        self, state: UniversalState, agent_output: Any
    ) -> UniversalAction:
        raise NotImplementedError

    @abstractmethod
    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> Any:
        raise NotImplementedError


@register_action_space()
class DefaultActionSpace(ActionSpace):
    """The standard action space: ``Discrete(13)`` covering indices 0–12.

    Agent outputs are raw integers that directly become action indices.
    This is the action space used by the paper and most models.
    """

    @property
    def gym_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(13)

    def agent_output_to_action(
        self, state: UniversalState, agent_output: int
    ) -> UniversalAction:
        return UniversalAction(action_idx=int(agent_output))

    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> int:
        return action.action_idx


@register_action_space()
class MinimalActionSpace(DefaultActionSpace):
    """A reduced action space without Tera: ``Discrete(9)`` covering indices 0–8.

    Tera move indices (9–12) coming from a dataset are mapped to regular move
    indices (0–3) by subtracting 9.  This allows training models that don't
    model the Tera gimmick on Gen 9 data.

    .. warning::
        ``action_to_agent_output`` mutates ``action.action_idx`` in place
        (subtracts 9), which may have side effects if the same ``UniversalAction``
        is reused.
    """

    @property
    def gym_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(9)

    def agent_output_to_action(
        self, state: UniversalState, agent_output: int
    ) -> UniversalAction:
        action_idx = int(agent_output)
        if action_idx >= 9:
            # map all gimmick move actions to regular move actions
            action_idx -= 9
        return UniversalAction(action_idx=action_idx)

    def action_to_agent_output(
        self, state: UniversalState, action: UniversalAction
    ) -> int:
        if action.action_idx >= 9:
            # map all gimmick move actions to regular move actions
            action.action_idx -= 9
        return action.action_idx


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    Subclasses implement ``__call__(last_state, state)`` which computes the
    reward for transitioning from ``last_state`` to ``state``.  The constructor
    accepts and ignores any keyword arguments so registries can instantiate
    reward functions uniformly.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        raise NotImplementedError


@register_reward_function()
class DefaultShapedReward(RewardFunction):
    """The default reward function from the paper.

    Reward components (all in [-1, 1] range except the terminal bonus):

    * **Damage dealt**: opponent HP decrease (0 to 1)
    * **HP healed / gained**: own active's HP increase (0 to 1)
    * **Status inflicted**: +0.5 for giving the opponent a status condition
    * **Status received**: -0.5 for taking a status condition
    * **Opponent Pokémon KO'd**: +1.0 when ``opponents_remaining`` decreases
    * **Own Pokémon KO'd**: -1.0 when ``available_switches`` shrinks
    * **Victory**: +100.0 (sparse terminal)
    * **Loss**: -100.0 (sparse terminal)

    The active Pokémon is matched across the transition by ``base_species``
    (not by unique ID), so it works even when the Pokémon object is replaced.
    If the active can't be matched (e.g. Revival Blessing brought back a
    previously-fainted mon), HP gain and status components are zeroed.
    """

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        active_now = state.player_active_pokemon
        active_prev = None
        for pokemon in [
            last_state.player_active_pokemon,
            *last_state.available_switches,
        ]:
            if pokemon.base_species == active_now.base_species:
                active_prev = pokemon
                break
        if active_prev is None:
            # this used to trigger a crash, but is now allowed because revival blessing in gen9 will break it
            hp_gain = 0.0
            took_status = 0.0
        else:
            hp_gain = active_now.hp_pct - active_prev.hp_pct
            took_status = float(
                active_now.status != "nostatus" and active_prev.status == "nostatus"
            )
        opp_now = state.opponent_active_pokemon
        opp_prev = last_state.opponent_active_pokemon
        if opp_now.base_species == opp_prev.base_species:
            damage_done = opp_prev.hp_pct - opp_now.hp_pct
            gave_status = float(
                opp_now.status != "nostatus" and opp_prev.status == "nostatus"
            )
        else:
            damage_done, gave_status = 0.0, 0.0
        lost_pokemon = float(
            len(last_state.available_switches) > len(state.available_switches)
        )
        removed_pokemon = float(
            last_state.opponents_remaining > state.opponents_remaining
        )
        if state.battle_won:
            victory = 1.0
        elif state.battle_lost:
            victory = -1.0
        else:
            victory = 0.0
        reward = (
            1.0 * (damage_done + hp_gain)
            + 0.5 * (gave_status - took_status)
            + 1.0 * (removed_pokemon - lost_pokemon)
            + 100.0 * victory
        )
        return reward


@register_reward_function()
class AggressiveShapedReward(RewardFunction):
    """A variant that removes loss penalty and increases win bonus to +200.

    The original policies had a tendency to cling to lost positions, dragging
    out unwinnable games.  This variant rewards only winning (0 for losing)
    and removes status-condition shaping to focus purely on HP and KO outcomes.
    The KO component is also doubled (2.0 vs 1.0).
    """

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        active_now = state.player_active_pokemon
        active_prev = None
        for pokemon in [
            last_state.player_active_pokemon,
            *last_state.available_switches,
        ]:
            if pokemon.base_species == active_now.base_species:
                active_prev = pokemon
                break
        hp_gain = 0.0 if active_prev is None else active_now.hp_pct - active_prev.hp_pct
        opp_now = state.opponent_active_pokemon
        opp_prev = last_state.opponent_active_pokemon
        if opp_now.base_species == opp_prev.base_species:
            damage_done = opp_prev.hp_pct - opp_now.hp_pct
        else:
            damage_done = 0.0
        lost_pokemon = float(
            len(last_state.available_switches) > len(state.available_switches)
        )
        removed_pokemon = float(
            last_state.opponents_remaining > state.opponents_remaining
        )
        victory = float(state.battle_won)
        reward = (
            1.0 * (damage_done + hp_gain)
            + 2.0 * (removed_pokemon - lost_pokemon)
            + 200.0 * victory
        )
        return reward


@register_reward_function()
class BinaryReward(RewardFunction):
    """A sparse-only variant: +100 for winning, -100 for losing, 0 otherwise.

    No shaping components — purely terminal reward.  Useful for credit
    assignment experiments and as a baseline for the shaped variants.
    """

    def __call__(self, last_state: UniversalState, state: UniversalState) -> float:
        if state.battle_won:
            return 100.0
        elif state.battle_lost:
            return -100.0
        return 0.0


class ObservationSpace(ABC):
    """Abstract interface for converting ``UniversalState`` to model observations.

    Each subclass defines:

    * ``gym_space`` — the Gymnasium observation space for the RL environment.
    * ``state_to_obs(state)`` — the core conversion logic.
    * ``tokenizable`` (optional) — which output keys are text that should be
      tokenized, and their expected (maximum) token length.
    * ``reset()`` (optional) — clear any history-dependent state between battles
      (e.g. accumulated revealed-opponent sets).

    The ``__call__`` method simply delegates to ``state_to_obs``.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.reset()

    def __name__(self) -> str:
        return self.__class__.__name__

    def reset(self) -> None:
        """Clear any internal state (between battles)."""

    @property
    def tokenizable(self) -> dict[str, int]:
        """Return a dictionary of tokenizable keys and their expected (max) length."""
        return {}

    @property
    @abstractmethod
    def gym_space(self) -> gym.spaces.Space:
        """Return the observation space for this observation type."""
        raise NotImplementedError

    @abstractmethod
    def state_to_obs(self, state: UniversalState) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def __call__(self, state: UniversalState) -> Dict[str, np.ndarray]:
        obs = self.state_to_obs(state)
        return obs


@register_observation_space()
class DefaultObservationSpace(ObservationSpace):
    """The default observation space from the paper.

    Produces a dictionary with two keys:

    * ``"numbers"`` — a ``(48,)`` float32 vector of numerical features:
      HP fractions, base stats ÷ 255, stat boosts ÷ 6, move base power,
      accuracy, priority, level, opponents remaining, etc.
    * ``"text"`` — a single string with ~87 whitespace-separated tokens
      encoding the format, player/opponent active Pokémon (species, item,
      ability, types, status, effect), 4 moves (name, type, category),
      up to 5 switches, conditions, weather, and previous moves.

    All Pokémon and move lists are sorted alphabetically via
    ``consistent_pokemon_order`` / ``consistent_move_order`` so the position
    of an entity in the observation is deterministic across backends.

    Padding: missing moves are filled with ``<blank>`` tokens; missing switches
    are filled with ``<blank>`` × N padding blocks.  Numerical padding uses -2.0
    (outside the normal [0,1] range) so models can learn to ignore it.

    Observation size: ``numbers.shape == (48,)``, ~87 text tokens.
    """

    @property
    def gym_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(48,),
                    dtype=np.float32,
                ),
                "text": gym.spaces.Text(
                    max_length=900,
                    min_length=800,
                    charset=set(string.ascii_lowercase)
                    | set(str(n) for n in range(0, 10))
                    | {"<", ">"},
                ),
            }
        )

    @property
    def tokenizable(self) -> dict[str, int]:
        return {
            "text": 87,
        }

    def _get_move_string_features(self, move: UniversalMove, active: bool) -> list[str]:
        """Text tokens for a move: name, and optionally type + category if active."""
        out = [clean_name(move.name)]
        if active:
            out += [clean_name(move.move_type), clean_name(move.category)]
        return out

    def _get_move_pad_string(self, active: bool) -> list[str]:
        """Blank text tokens to fill a missing move slot."""
        out = ["<blank>"]
        if active:
            out += ["<blank>", "<blank>"]
        return out

    def _get_move_numerical_features(
        self, move: UniversalMove, active: bool
    ) -> list[float]:
        """Numerical features for a move: base_power/200, accuracy, priority/5.

        Returns empty list if the move is not active (bench/fainted Pokémon).
        """
        if not active:
            return []
        return [move.base_power / 200.0, move.accuracy, move.priority / 5.0]

    def _get_move_pad_numerical(self, active: bool) -> list[float]:
        """Numerical padding for missing move slots: ``[-2.0, -2.0, -2.0]``."""
        if not active:
            return []
        return [-2.0] * 3

    def _get_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        """Text tokens for a Pokémon: name, item, ability.

        If active: adds types, effect, status. If bench: adds ``<moveset>``
        tag followed by (possibly blank-padded) move names.
        """
        out = [pokemon.name, pokemon.item, pokemon.ability]
        if active:
            out += [pokemon.types, pokemon.effect, pokemon.status]
        else:
            out += ["<moveset>"]
            move_num = -1
            for move_num, move in enumerate(consistent_move_order(pokemon.moves)):
                out += self._get_move_string_features(move, active=False)
            while move_num < 3:
                out += self._get_move_pad_string(active=False)
                move_num += 1
        return out

    def _get_opponent_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        return self._get_pokemon_string_features(pokemon, active)

    def _get_pokemon_pad_string(self, active: bool) -> list[str]:
        """Blank text tokens to pad a missing Pokémon slot."""
        blanks = 3 + (4 if active else 5)
        return ["<blank>"] * blanks

    def _get_pokemon_numerical_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[float]:
        """Numerical features for a Pokémon: HP fraction, level/100, base stats/255,
        and (if active) boost stages/6 for all 7 stats."""
        out = [pokemon.hp_pct]
        if active:
            stat = lambda s: getattr(pokemon, f"base_{s}") / 255.0
            boost = lambda b: getattr(pokemon, f"{b}_boost") / 6.0
            out.append(pokemon.lvl / 100.0)
            out += map(stat, ["atk", "spa", "def", "spd", "spe", "hp"])
            out += map(
                boost, ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            )
        return out

    def _get_pokemon_pad_numerical(self, active: bool) -> list[float]:
        """Numerical padding for missing Pokémon slots."""
        blanks = 1 + (14 if active else 0)
        return [-2.0] * blanks

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        """Convert a UniversalState to the default text+numbers observation.

        Builds the observation by walking through a fixed sequence of entity blocks:
        ``<player>``, ``<move>`` ×4, ``<switch>`` ×5, ``<opponent>``,
        ``<conditions>``, ``<player_prev>``, ``<opp_prev>``.  All Pokémon and
        moves are sorted alphabetically for determinism.  Missing switches/moves
        are padded with ``<blank>`` tokens and -2.0 numerical features.
        """
        player_str = ["<player>"] + self._get_pokemon_string_features(
            state.player_active_pokemon, active=True
        )
        numerical = [
            state.opponents_remaining / 6.0
        ] + self._get_pokemon_numerical_features(
            state.player_active_pokemon, active=True
        )

        # consistent move order
        move_str, move_num = [], -1
        for move_num, move in enumerate(
            consistent_move_order(state.player_active_pokemon.moves)
        ):
            move_str += ["<move>"] + self._get_move_string_features(move, active=True)
            numerical += self._get_move_numerical_features(move, active=True)

        while move_num < 3:
            move_str += ["<move>"] + self._get_move_pad_string(active=True)
            numerical += self._get_move_pad_numerical(active=True)
            move_num += 1

        # consistent switch order
        switch_str, switch_num = [], -1
        for switch_num, switch in enumerate(
            consistent_pokemon_order(state.available_switches)
        ):
            switch_str += ["<switch>"] + self._get_pokemon_string_features(
                switch, active=False
            )
            numerical += self._get_pokemon_numerical_features(switch, active=False)
        while switch_num < 4:
            switch_str += ["<switch>"] + self._get_pokemon_pad_string(active=False)
            numerical += self._get_pokemon_pad_numerical(active=False)
            switch_num += 1

        force_switch = "<forcedswitch>" if state.forced_switch else "<anychoice>"
        opponent_str = ["<opponent>"] + self._get_opponent_pokemon_string_features(
            state.opponent_active_pokemon, active=True
        )
        numerical += self._get_pokemon_numerical_features(
            state.opponent_active_pokemon, active=True
        )
        global_str = ["<conditions>"] + [
            state.weather,
            state.player_conditions,
            state.opponent_conditions,
        ]
        prev_move_str = (
            ["<player_prev>"]
            + self._get_move_string_features(state.player_prev_move, active=False)
            + ["<opp_prev>"]
            + self._get_move_string_features(state.opponent_prev_move, active=False)
        )
        full_text_list = (
            [f"<{state.agent_format}>", force_switch]
            + player_str
            + move_str
            + switch_str
            + opponent_str
            + global_str
            + prev_move_str
        )
        # length should be 85 (type features have 2 words --> final word length of 87)
        text = " ".join(full_text_list)
        text = np.array(text, dtype=np.str_)
        numbers = np.array(numerical, dtype=np.float32)
        return {"text": text, "numbers": numbers}


@register_observation_space()
class ExpandedObservationSpace(DefaultObservationSpace):
    """Adds PP, the opponent's revealed party, and edge case sleep/freeze flags to DefaultObservationSpace.

    The DefaultObservationSpace used by the paper makes Pokémon more long-term-memory-intensive
    than it strictly needs to be:

    1. Sleep/freeze clause relies on remembering our move and the opponent active Pokémon's status
        at previous timesteps.

    2. PP counts can only be inferred by recalling prev_move features at previous timesteps.

    3. The opponent's full team must be inferred from recalling the active Pokémon at previous
        timesteps.

    This observation space moves some of that information into every timestep. Also adds tera types for gen 9.
    """

    def reset(self) -> None:
        """Reset the history-dependent state at the start of each battle."""
        self.any_opponent_asleep = False
        self.any_opponent_frozen = False
        self.revealed_opponents = set()

    @property
    def gym_space(self) -> gym.spaces.Dict:
        base_space = super().gym_space
        base_space["numbers"] = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            # adds 4 PP features + 2 sleep/freeze flags + 1 can_tera flag
            shape=(48 + 7,),
            dtype=np.float32,
        )
        return base_space

    @property
    def tokenizable(self) -> dict[str, int]:
        # adds 6 new tokens for the revealed party
        # adds 6 new tokens for the tera types of our party, 1 for the opponent
        return {"text": 87 + 13}

    def _get_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        base = super()._get_pokemon_string_features(pokemon, active)
        base.append(pokemon.tera_type)
        return base

    def _get_pokemon_pad_string(self, active: bool) -> list[str]:
        blanks = 4 + (4 if active else 5)
        return ["<blank>"] * blanks

    def _get_move_numerical_features(
        self, move: UniversalMove, active: bool
    ) -> list[float]:
        """Extends the default with a discretized PP warning feature.

        PP tracking in replays is approximate (off-by-one errors from PP Ups,
        Mimic-inherited PP, etc.).  Instead of a raw PP ratio, this emits a
        discretized 'PP warning' in {0, 1, 2, 3}:

        * 0 → PP ratio = 0 (move depleted)
        * 1 → 0 < ratio < 0.25 (critical — 1 or 2 uses left)
        * 2 → 0.25 ≤ ratio < 0.5 (low)
        * 3 → ratio ≥ 0.5 (healthy)
        """
        out = super()._get_move_numerical_features(move, active)
        if active:
            pp_ratio = move.current_pp / move.max_pp
            # there's a reason the original obs space doesn't have PP counts ---
            # they are not accurate in replays. Compromise by discretizing to
            # "low pp" warnings that would minimize off-by-one shift:
            pp_warning = (pp_ratio >= 0.5) + (pp_ratio >= 0.25) + (pp_ratio > 0)
            out.append(float(pp_warning))
        return out

    def _get_move_pad_numerical(self, active: bool) -> list[float]:
        if not active:
            return []
        return [-2.0] * 4

    def state_to_obs(self, state: UniversalState) -> Dict[str, np.ndarray]:
        """Build observation with PP, revealed-opponent history, and sleep/freeze flags.

        Extends the default observation by appending:

        * Accumulated ``any_opponent_asleep`` / ``any_opponent_frozen`` booleans
          (needed because sleep/freeze clause depends on whether the opponent
          has *ever* been put to sleep/frozen by you).
        * ``can_tera`` flag.
        * Sorted list of all opponent species revealed so far (padded to 6 with
          ``<blank>``).
        """
        obs = super().state_to_obs(state)

        opponent = state.opponent_active_pokemon
        # (sleep/freeze clause only activates when *we* put the opponent to sleep/freeze,
        # which is not what's being tracked here, but this covers the main failure case
        # and the subtlety has been learnable without this feature.)
        self.any_opponent_asleep |= opponent.status == "slp"
        self.any_opponent_frozen |= opponent.status == "frz"
        new_features = [
            self.any_opponent_asleep,
            self.any_opponent_frozen,
            state.can_tera,
        ]
        obs["numbers"] = np.concatenate([obs["numbers"], new_features])

        # add a list of revealed opponents padded to length 6 while reusing
        # the existing <blank> token to avoid making a new vocabulary.
        self.revealed_opponents.add(opponent.base_species)
        revealed = [opp_name for opp_name in sorted(self.revealed_opponents)]
        while len(revealed) < 6:
            revealed.append("<blank>")
        obs["text"] = np.array(
            obs["text"].item() + " " + " ".join(revealed[:6]), dtype=np.str_
        )
        return obs


@register_observation_space()
class TeamPreviewObservationSpace(ExpandedObservationSpace):
    """Extends ``ExpandedObservationSpace`` with opponent team preview info.

    Appends the 6 opponent species shown during team preview (sorted, padded
    with ``<blank>``).  This gives the agent complete knowledge of the
    opponent's possible Pokémon from turn 1, matching what a human player sees.
    """

    @property
    def tokenizable(self) -> dict[str, int]:
        # adds 6 new tokens for teampreview
        return {"text": 87 + 13 + 6}

    def state_to_obs(self, state: UniversalState) -> Dict[str, np.ndarray]:
        """Build observation including team preview species."""
        obs = super().state_to_obs(state)
        teampreview = [opp_name for opp_name in sorted(state.opponent_teampreview)]
        while len(teampreview) < 6:
            teampreview.append("<blank>")
        obs["text"] = np.array(
            obs["text"].item() + " " + " ".join(teampreview[:6]), dtype=np.str_
        )
        return obs


@register_observation_space()
class OpponentMoveObservationSpace(TeamPreviewObservationSpace):
    """Trades move-category tokens to make room for the opponent's revealed moves.

    Drops the move category string from our own active moves (saving 4 tokens)
    and uses the budget to include the names of any opponent moves that have
    been revealed so far (up to 4, ``<blank>``-padded).
    """

    def _get_move_string_features(self, move: UniversalMove, active: bool) -> list[str]:
        out = [clean_name(move.name)]
        if active:
            # save 4 tokens
            out += [clean_name(move.move_type)]
        return out

    def _get_move_pad_string(self, active: bool) -> list[str]:
        return ["<blank>"] * (2 if active else 1)

    def _get_opponent_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        base = self._get_pokemon_string_features(pokemon, active)
        # add 4 tokens
        moves = ["<blank>"] * 4
        for i, move in enumerate(consistent_move_order(pokemon.moves)[:4]):
            moves[i] = clean_name(move.name)
        return base + moves


@register_observation_space()
class GroupedObservationSpace(ObservationSpace):
    """
    Groups observations by entity for use with a shared Pokemon encoder.

    Unlike DefaultObservationSpace which concatenates all features into single
    "text" and "numbers" arrays, this space outputs separate arrays for each
    Pokemon and a misc array for global state.
    """

    POKEMON_TEXT_LEN = 12  # name, item, ability, tera, types×2, effect, status, moves×4
    POKEMON_NUM_LEN = 31  # hp, lvl, stats×6, boosts×7, (bp, acc, pri, pp)×4
    MISC_TEXT_LEN = (
        20  # format, switch, weather, field, conds×2, prev×2, revealed×6, preview×6
    )
    MISC_NUM_LEN = 4  # opp_remaining, sleep, freeze, can_tera
    NUM_SWITCHES = 5

    def reset(self) -> None:
        self.any_opponent_asleep = False
        self.any_opponent_frozen = False
        self.revealed_opponents = set()

    @property
    def tokenizable(self) -> dict[str, int]:
        return {
            "text_active_pokemon": self.POKEMON_TEXT_LEN,
            "text_switch_0": self.POKEMON_TEXT_LEN,
            "text_switch_1": self.POKEMON_TEXT_LEN,
            "text_switch_2": self.POKEMON_TEXT_LEN,
            "text_switch_3": self.POKEMON_TEXT_LEN,
            "text_switch_4": self.POKEMON_TEXT_LEN,
            "text_opponent_active_pokemon": self.POKEMON_TEXT_LEN,
            "text_misc": self.MISC_TEXT_LEN,
        }

    @property
    def gym_space(self) -> gym.spaces.Dict:
        spaces = {}
        for key in self.tokenizable:
            spaces[key] = gym.spaces.Text(max_length=500, min_length=0)
        spaces["numbers_active_pokemon"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
        )
        for i in range(self.NUM_SWITCHES):
            spaces[f"numbers_switch_{i}"] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
            )
        spaces["numbers_opponent_active_pokemon"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.POKEMON_NUM_LEN,), dtype=np.float32
        )
        spaces["numbers_misc"] = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.MISC_NUM_LEN,), dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    def _get_universal_pokemon_text(
        self, pokemon: UniversalPokemon, is_active: bool = True
    ) -> list[str]:
        out = [
            pokemon.name,
            pokemon.item,
            pokemon.ability,
            pokemon.tera_type,
        ]
        type_parts = pokemon.types.split()
        out.extend(type_parts[:2] + ["notype"] * (2 - len(type_parts)))
        out.append(pokemon.effect)
        out.append(pokemon.status)
        # (sorted order, padded to 4)
        for move in self._pad_moves(pokemon.moves):
            out.append(clean_name(move.name) if move else "<blank>")

        return out

    def _get_universal_pokemon_numbers(
        self, pokemon: UniversalPokemon, is_active: bool = True
    ) -> list[float]:
        out = [pokemon.hp_pct, pokemon.lvl / 100.0]
        for stat in ("atk", "spa", "def", "spd", "spe", "hp"):
            out.append(getattr(pokemon, f"base_{stat}") / 255.0)
        if is_active:
            for boost in ("atk", "spa", "def", "spd", "spe", "accuracy", "evasion"):
                out.append(getattr(pokemon, f"{boost}_boost") / 6.0)
        else:
            out.extend([0.0] * 7)
        # (sorted order, padded to 4)
        for move in self._pad_moves(pokemon.moves):
            if move:
                pp_ratio = move.current_pp / move.max_pp if move.max_pp > 0 else 0.0
                pp_warning = (pp_ratio >= 0.5) + (pp_ratio >= 0.25) + (pp_ratio > 0)
                out.extend(
                    [
                        move.base_power / 200.0,
                        move.accuracy,
                        move.priority / 5.0,
                        float(pp_warning),
                    ]
                )
            else:
                out.extend([-2.0] * 4)

        return out

    def _pad_moves(
        self, moves: list[UniversalMove], n: int = 4
    ) -> list[Optional[UniversalMove]]:
        sorted_moves = consistent_move_order(moves)[:n]
        return sorted_moves + [None] * (n - len(sorted_moves))

    def _get_blank_pokemon_text(self) -> list[str]:
        """Return a list of ``<blank>`` tokens for an empty Pokémon slot."""
        return ["<blank>"] * self.POKEMON_TEXT_LEN

    def _get_blank_pokemon_numbers(self) -> list[float]:
        """Return a list of -2.0 values for an empty Pokémon numerical slot."""
        return [-2.0] * self.POKEMON_NUM_LEN

    def _get_misc_text(self, state: UniversalState) -> list[str]:
        """Build the text feature list for global (non-Pokémon) state.

        Includes: format tag, forced-switch flag, weather, battle field,
        side conditions, previous moves, revealed opponent species, and
        team preview.  ``"nofield"`` is mapped to ``<blank>`` because it
        isn't in the tokenizer vocabulary.
        """
        battle_field = (
            state.battle_field if state.battle_field != "nofield" else "<blank>"
        )
        out = [
            f"<{state.agent_format}>",
            "<forcedswitch>" if state.forced_switch else "<anychoice>",
            state.weather,
            battle_field,
            state.player_conditions,
            state.opponent_conditions,
            clean_name(state.player_prev_move.name),
            clean_name(state.opponent_prev_move.name),
        ]
        revealed = sorted(self.revealed_opponents)[:6]
        out.extend(revealed + ["<blank>"] * (6 - len(revealed)))
        teampreview = sorted(state.opponent_teampreview)[:6]
        out.extend(teampreview + ["<blank>"] * (6 - len(teampreview)))
        return out

    def _get_misc_numbers(self, state: UniversalState) -> list[float]:
        """Build the numerical feature list for global state.

        Includes: opponents_remaining/6, any_opponent_asleep flag,
        any_opponent_frozen flag, can_tera flag.
        """
        return [
            state.opponents_remaining / 6.0,
            float(self.any_opponent_asleep),
            float(self.any_opponent_frozen),
            float(state.can_tera),
        ]

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        obs = {}

        # update history-dependent state tracking
        opponent = state.opponent_active_pokemon
        self.any_opponent_asleep |= opponent.status == "slp"
        self.any_opponent_frozen |= opponent.status == "frz"
        self.revealed_opponents.add(opponent.base_species)

        # player active
        obs["text_active_pokemon"] = self._get_universal_pokemon_text(
            state.player_active_pokemon, is_active=True
        )
        obs["numbers_active_pokemon"] = self._get_universal_pokemon_numbers(
            state.player_active_pokemon, is_active=True
        )

        # reserve team (sorted order, padded to NUM_SWITCHES)
        switches = consistent_pokemon_order(state.available_switches)
        for i in range(self.NUM_SWITCHES):
            if i < len(switches):
                obs[f"text_switch_{i}"] = self._get_universal_pokemon_text(
                    switches[i], is_active=False
                )
                obs[f"numbers_switch_{i}"] = self._get_universal_pokemon_numbers(
                    switches[i], is_active=False
                )
            else:
                obs[f"text_switch_{i}"] = self._get_blank_pokemon_text()
                obs[f"numbers_switch_{i}"] = self._get_blank_pokemon_numbers()

        # opponent active
        obs["text_opponent_active_pokemon"] = self._get_universal_pokemon_text(
            state.opponent_active_pokemon, is_active=True
        )
        obs["numbers_opponent_active_pokemon"] = self._get_universal_pokemon_numbers(
            state.opponent_active_pokemon, is_active=True
        )

        # misc (global state)
        obs["text_misc"] = self._get_misc_text(state)
        obs["numbers_misc"] = self._get_misc_numbers(state)

        # temporary assert checks to verify lengths
        for key in obs:
            if key.startswith("text"):
                expected = (
                    self.MISC_TEXT_LEN if key == "text_misc" else self.POKEMON_TEXT_LEN
                )
                assert (
                    len(obs[key]) == expected
                ), f"{key}: expected {expected}, got {len(obs[key])}"
            else:
                expected = (
                    self.MISC_NUM_LEN if key == "numbers_misc" else self.POKEMON_NUM_LEN
                )
                assert (
                    len(obs[key]) == expected
                ), f"{key}: expected {expected}, got {len(obs[key])}"

        for key in obs:
            if key.startswith("text"):
                obs[key] = np.array(" ".join(obs[key]), dtype=np.str_)
            else:
                obs[key] = np.array(obs[key], dtype=np.float32)

        return obs


@register_observation_space()
class PatchPokeAgentTeraBug(ObservationSpace):
    """Wrapper that intentionally reintroduces a tera-type bug for PokeAgent compatibility.

    The "pokeagent" backend had a bug where tera_type was never set, so models
    trained with that backend learned to expect ``"notype"`` for all tera fields.
    This wrapper patches the state by setting all player tera types to ``"notype"``
    before delegating to the wrapped observation space, so those models can still
    make sound decisions when run with the metamon backend.
    """

    def __init__(self, base_obs_space: ObservationSpace) -> None:
        self.base_obs_space = base_obs_space
        super().__init__()

    def reset(self) -> None:
        self.base_obs_space.reset()

    @property
    def gym_space(self) -> gym.spaces.Space:
        return self.base_obs_space.gym_space

    @property
    def tokenizable(self) -> Dict[str, int]:
        return self.base_obs_space.tokenizable

    def state_to_obs(self, state: UniversalState) -> Dict[str, np.ndarray]:
        patched_state = copy.deepcopy(state)
        # patch player pokemon tera types to "notype"
        patched_state.player_active_pokemon.tera_type = "notype"
        for pokemon in patched_state.available_switches:
            pokemon.tera_type = "notype"
        return self.base_obs_space.state_to_obs(patched_state)


# Register patched versions of observation spaces for PokeAgent Challenge compatibility
@register_observation_space("PAC-ExpandedObservationSpace")
class PACExpandedObservationSpace(PatchPokeAgentTeraBug):
    """PAC-compatible wrapper around ``ExpandedObservationSpace``."""
    def __init__(self) -> None:
        super().__init__(ExpandedObservationSpace())


@register_observation_space("PAC-TeamPreviewObservationSpace")
class PACTeamPreviewObservationSpace(PatchPokeAgentTeraBug):
    """PAC-compatible wrapper around ``TeamPreviewObservationSpace``."""
    def __init__(self) -> None:
        super().__init__(TeamPreviewObservationSpace())


@register_observation_space("PAC-OpponentMoveObservationSpace")
class PACOpponentMoveObservationSpace(PatchPokeAgentTeraBug):
    """PAC-compatible wrapper around ``OpponentMoveObservationSpace``."""
    def __init__(self) -> None:
        super().__init__(OpponentMoveObservationSpace())


class TokenizedObservationSpace(ObservationSpace):
    """An observation space that tokenizes specified keys of the default observation space.

    Splits text into whitespace-separated words and runs them through a simple
    vocabulary lookup, which usually has been generated by tracking unique words across
    the entire replay dataset. Useful for turning the text features of the default
    observation space into an array with constant shape.
    """

    def __init__(
        self,
        base_obs_space: ObservationSpace,
        tokenizer: PokemonTokenizer,
    ) -> None:
        self.base_obs_space = base_obs_space
        self.tokenizer = tokenizer

    def reset(self) -> None:
        self.base_obs_space.reset()

    @property
    def gym_space(self) -> gym.spaces.Dict:
        tokenizable = self.base_obs_space.tokenizable
        base_space = copy.deepcopy(self.base_obs_space.gym_space)
        new_space_dict = {
            key: space
            for key, space in base_space.spaces.items()
            if key not in tokenizable
        }
        for tokenizable_key, tokenizable_length in tokenizable.items():
            low_token = min(UNKNOWN_TOKEN, 0)
            high_token = max(UNKNOWN_TOKEN, len(self.tokenizer))
            new_space_dict[f"{tokenizable_key}_tokens"] = gym.spaces.Box(
                low=low_token,
                high=high_token,
                shape=(tokenizable_length,),
                dtype=np.int32,
            )

        return gym.spaces.Dict(new_space_dict)

    def state_to_obs(self, state: UniversalState) -> Dict[str, np.ndarray]:
        """Build base observation, then tokenize all keys listed in ``tokenizable``.

        For each tokenizable key ``K``, the original text array is popped from
        the observation dict and replaced with ``K_tokens`` — an int32 array
        of vocabulary indices.
        """
        obs = self.base_obs_space.state_to_obs(state)
        for tokenizable_key in self.base_obs_space.tokenizable.keys():
            base_obs_key = obs.pop(tokenizable_key)
            obs[f"{tokenizable_key}_tokens"] = self.tokenizer.tokenize(
                base_obs_key.tolist()
            )
        return obs


@register_observation_space()
class WorldModelObservationSpace(ObservationSpace):
    """Observation space for world model training.

    Extends the default observation space with:
        - HP (0.0-1.0 float or "unknown"), split into space-separated characters
          (e.g. ``0 . 7 3``), after every Pokémon name
        - Opponent bench information with ``<opponent_switch>`` blocks
        - Player fainted Pokémon with ``<fainted>`` blocks
        - Opponent fainted Pokémon with ``<opponent_fainted>`` blocks
        - ``<boosts>`` block with stat stage changes (e.g. ``spa-1``, ``atk+2``)
          for every Pokémon in play — critical for world model state tracking
        - ``unknownability``, ``unknownitem`` tokens for unrevealed info
        - ``<opponent_moveset>`` tag for opponent movesets (no ``unknownmove`` fillers)
        - Terminal token: ``<ongoing>`` / ``<won>`` / ``<lost>``

    **No padding** — ``<switch>``, ``<opponent_switch>``, ``<fainted>``, and
    ``<opponent_fainted>`` blocks are only emitted when they actually contain
    Pokémon.  Opponent movesets only emit revealed moves (no ``unknownmove``
    fillers).  ``<boosts>`` emits ``none`` when no stat changes are active,
    or individual tokens like ``spa-1``, ``atk+2`` for non-zero stages.
    This eliminates redundant identical-token computation during training.

    Text format (~52–312 tokens; all repeated blocks are variable-length):
        ``<format> <forcedswitch|anychoice>``
        ``<player> <name> <hp_c0>..<hp_c3> <item> <ability> <type_0> <type_1> <effect> <status> <boosts> <boost...>``
        ``<move> <name> <type> <category>``  ×4
        ``<switch> <name> <hp_c0>..<hp_c3> <item> <ability> <boosts> <boost...> <moveset>``
        ``  <move_1>..<move_4>``  ×0–5
        ``<opponent> <name> <hp_c0>..<hp_c3> <item> <ability> <type_0> <type_1> <effect> <status> <boosts> <boost...> <opponent_moveset>``
        ``  <move_1>..<move_N>``  (0–4 revealed moves, no fillers)
        ``<opponent_switch> <name> <hp_c0>..<hp_c3> <item> <ability> <status> <effect> <boosts> <boost...> <opponent_moveset>``
        ``  <move_1>..<move_N>``  ×0–5 (0–4 revealed moves each, no fillers)
        ``<fainted> <name> <hp_c0>..<hp_c3> <item> <ability> <boosts> <boost...> <moveset>``
        ``  <move_1>..<move_4>``  ×0–5 (only actual fainted Pokémon)
        ``<opponent_fainted> <name> <hp_c0>..<hp_c3> <item> <ability> <status> <effect> <boosts> <boost...> <opponent_moveset>``
        ``  <move_1>..<move_N>``  ×0–5 (only actual fainted Pokémon, 0–4 revealed moves)
        ``<conditions> <weather> <player_cond> <opponent_cond>``
        ``<player_prev> <move> <opp_prev> <move>``
        ``<ongoing|won|lost>``
    """

    NUM_FAINTED_SLOTS = 5
    NUM_OPPONENT_FAINTED_SLOTS = 5

    @property
    def gym_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(63,),  # 48 orig + 5 opp bench + 5 fainted + 5 opp fainted
                    dtype=np.float32,
                ),
                "text": gym.spaces.Text(
                    max_length=2500,
                    min_length=500,
                    charset=set(string.ascii_lowercase)
                    | set(str(n) for n in range(0, 10))
                    | {"<", ">"},
                ),
            }
        )

    @property
    def tokenizable(self) -> dict[str, int]:
        # soft max — no padding for <switch> / <opponent_switch> /
        # <fainted> / <opponent_fainted> blocks; opponent moveset blocks
        # only emit revealed moves (no unknownmove fillers).
        # Maximum token count occurs when all 12 Pokémon are fainted / on
        # field with full revealed movesets.
        return {
            "text": 312,
        }

    @staticmethod
    def _hp_str(hp_pct: float) -> list[str]:
        """Format HP as space-separated fixed-point characters: ``1 . 0 0`` or ``unknown``.

        Each character (digit or dot) becomes a separate token so the tokenizer
        only needs 0-9 and ``.`` for HP values.  This avoids needing a unique
        vocabulary entry for every possible HP fraction.
        """
        if hp_pct is None:
            return ["unknown"]
        # Format as fixed-point with exactly 2 decimals, then split into characters
        formatted = f"{hp_pct:.2f}"  # e.g. "1.00", "0.73", "0.00"
        return list(formatted)

    # Stat boost attributes (cross-generational — Gen 1 only uses atk/def/spa/spd/spe;
    # accuracy/evasion appear in later gens).  UniversalPokemon always carries all of them.
    _BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

    @classmethod
    def _boost_tokens(cls, pokemon) -> list[str]:
        """Return ``<boosts>`` token list for a Pokémon.

        Emits ``["<boosts>", "none"]`` if all boosts are zero, otherwise
        ``["<boosts>", "atk+1", "spa-1", ...]`` with one token per non-zero boost.
        Boost tokens look like ``atk+1``, ``spa-1``, ``spe+2`` (no leading zeros).

        This is critical for world model state tracking: the model needs to
        observe stat changes as discrete tokens to predict future boost states.
        """
        tokens = []
        for stat in cls._BOOST_STATS:
            val = getattr(pokemon, f"{stat}_boost", 0)
            if val != 0:
                sign = "+" if val > 0 else ""
                tokens.append(f"{stat}{sign}{val}")
        if tokens:
            return ["<boosts>"] + tokens
        return ["<boosts>", "none"]

    def _get_move_string_features(self, move: UniversalMove, active: bool) -> list[str]:
        """Text tokens for a move: name, and optionally type + category if active."""
        out = [clean_name(move.name)]
        if active:
            out += [clean_name(move.move_type), clean_name(move.category)]
        return out

    def _get_move_pad_string(self, active: bool, use_unknownmove: bool = False) -> list[str]:
        """Blank text tokens to fill a missing move slot.

        Uses ``unknownmove`` as the pad token when ``use_unknownmove=True``
        (for opponent movesets with unrevealed slots), otherwise ``<blank>``.
        """
        pad_token = "unknownmove" if use_unknownmove else "<blank>"
        out = [pad_token]
        if active:
            out += ["<blank>", "<blank>"]
        return out

    def _get_move_numerical_features(
        self, move: UniversalMove, active: bool
    ) -> list[float]:
        """Numerical features for a move: base_power/200, accuracy, priority/5."""
        if not active:
            return []
        return [move.base_power / 200.0, move.accuracy, move.priority / 5.0]

    def _get_move_pad_numerical(self, active: bool) -> list[float]:
        """Numerical padding for missing move slots."""
        if not active:
            return []
        return [-2.0] * 3

    def _get_pokemon_string_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[str]:
        """Base string features shared by all Pokémon blocks.

        Does NOT include types/effect/status (active-only) or moveset/moves —
        those are added by callers depending on context (player/opponent/bench).
        Includes HP as space-separated digits and boost tokens.
        """
        out = [pokemon.name] + self._hp_str(pokemon.hp_pct) + [pokemon.item, pokemon.ability]
        out += self._boost_tokens(pokemon)
        return out

    def _get_player_active_string_features(self, pokemon: UniversalPokemon) -> list[str]:
        """Full string features for the player's active Pokémon.

        Includes: name, HP digits, item, ability, types, effect, status, boosts.
        """
        return (
            [pokemon.name]
            + self._hp_str(pokemon.hp_pct)
            + [pokemon.item, pokemon.ability, pokemon.types, pokemon.effect, pokemon.status]
            + self._boost_tokens(pokemon)
        )

    def _get_player_bench_string_features(self, pokemon: UniversalPokemon) -> list[str]:
        """Full string features for a player bench Pokémon.

        Includes 4 moves (``<blank>``-padded) after a ``<moveset>`` tag.
        """
        out = (
            [pokemon.name]
            + self._hp_str(pokemon.hp_pct)
            + [pokemon.item, pokemon.ability]
            + self._boost_tokens(pokemon)
            + ["<moveset>"]
        )
        move_num = -1
        for move_num, move in enumerate(consistent_move_order(pokemon.moves)):
            out += self._get_move_string_features(move, active=False)
        while move_num < 3:
            out += self._get_move_pad_string(active=False)
            move_num += 1
        return out

    def _get_player_fainted_string_features(self, pokemon: UniversalPokemon) -> list[str]:
        """Full string features for a player fainted Pokémon.

        Same format as bench (name, HP, item, ability, boosts, moveset + 4 moves).
        """
        return self._get_player_bench_string_features(pokemon)

    def _get_opponent_active_string_features(self, pokemon: UniversalPokemon) -> list[str]:
        """Full string features for the opponent's active Pokémon.

        Uses ``<opponent_moveset>`` tag and only emits revealed moves — no
        ``unknownmove`` fillers for unrevealed slots.  This avoids redundant
        computation on tokens the model knows are meaningless.
        """
        base = (
            [pokemon.name]
            + self._hp_str(pokemon.hp_pct)
            + [pokemon.item, pokemon.ability, pokemon.types, pokemon.effect, pokemon.status]
            + self._boost_tokens(pokemon)
        )
        # Only emit revealed moves — no unknownmove fillers.
        moves = []
        for move in consistent_move_order(pokemon.moves)[:4]:
            moves.append(clean_name(move.name))
        return base + ["<opponent_moveset>"] + moves

    def _get_opponent_inactive_string_features(self, pokemon: UniversalPokemon) -> list[str]:
        """Full string features for an opponent bench / fainted Pokémon.

        Includes status and effect (unlike player bench).  Only emits revealed
        moves (no ``unknownmove`` fillers).
        """
        out = (
            [pokemon.name]
            + self._hp_str(pokemon.hp_pct)
            + [pokemon.item, pokemon.ability, pokemon.status, pokemon.effect]
            + self._boost_tokens(pokemon)
            + ["<opponent_moveset>"]
        )
        for move in consistent_move_order(pokemon.moves):
            out += self._get_move_string_features(move, active=False)
        return out

    def _get_pokemon_numerical_features(
        self, pokemon: UniversalPokemon, active: bool
    ) -> list[float]:
        """Numerical features for a Pokémon: HP fraction, plus (if active)
        level/100, base stats/255, and boost stages/6."""
        out = [pokemon.hp_pct]
        if active:
            stat = lambda s: getattr(pokemon, f"base_{s}") / 255.0
            boost = lambda b: getattr(pokemon, f"{b}_boost") / 6.0
            out.append(pokemon.lvl / 100.0)
            out += map(stat, ["atk", "spa", "def", "spd", "spe", "hp"])
            out += map(
                boost, ["atk", "spa", "def", "spd", "spe", "accuracy", "evasion"]
            )
        return out

    def _get_pokemon_pad_numerical(self, active: bool) -> list[float]:
        """Numerical padding for missing Pokémon slots."""
        blanks = 1 + (14 if active else 0)
        return [-2.0] * blanks

    def state_to_obs(self, state: UniversalState) -> dict[str, np.ndarray]:
        """Convert a UniversalState to the world-model observation format.

        Builds a variable-length text string (~52–312 tokens) with entity blocks
        for the player active, 4 moves, player bench (0–5), opponent active,
        opponent bench (0–5), player fainted (0–5), opponent fainted (0–5),
        conditions, previous moves, and a terminal token (``<ongoing>``,
        ``<won>``, or ``<lost>``).

        Unlike ``DefaultObservationSpace``, text blocks for bench/fainted
        Pokémon are only emitted when they actually exist — no padding.
        Numerical features are still padded to fixed shapes (5 slots each for
        opponent bench, fainted, and opponent fainted).
        """
        # ── player active ──
        player_str = ["<player>"] + self._get_player_active_string_features(
            state.player_active_pokemon
        )
        numerical = [
            state.opponents_remaining / 6.0
        ] + self._get_pokemon_numerical_features(
            state.player_active_pokemon, active=True
        )

        # player moves (4 × <move> name type category, padded with <blank>)
        move_str, move_num = [], -1
        for move_num, move in enumerate(
            consistent_move_order(state.player_active_pokemon.moves)
        ):
            move_str += ["<move>"] + self._get_move_string_features(move, active=True)
            numerical += self._get_move_numerical_features(move, active=True)
        while move_num < 3:
            move_str += ["<move>"] + self._get_move_pad_string(active=True)
            numerical += self._get_move_pad_numerical(active=True)
            move_num += 1

        # ── player bench (variable-length text, fixed numerical padding) ──
        switch_str = []
        switch_count = 0
        for switch in consistent_pokemon_order(state.available_switches):
            switch_str += ["<switch>"] + self._get_player_bench_string_features(switch)
            numerical += self._get_pokemon_numerical_features(switch, active=False)
            switch_count += 1
        while switch_count < 5:
            numerical += self._get_pokemon_pad_numerical(active=False)
            switch_count += 1

        # ── opponent active ──
        force_switch = "<forcedswitch>" if state.forced_switch else "<anychoice>"
        opponent_str = ["<opponent>"] + self._get_opponent_active_string_features(
            state.opponent_active_pokemon
        )
        numerical += self._get_pokemon_numerical_features(
            state.opponent_active_pokemon, active=True
        )

        # ── opponent bench (variable-length text, fixed numerical padding) ──
        opponent_bench_str = []
        opp_bench_count = 0
        for bench_poke in consistent_pokemon_order(state.opponent_bench):
            opponent_bench_str += ["<opponent_switch>"] + self._get_opponent_inactive_string_features(
                bench_poke
            )
            numerical += self._get_pokemon_numerical_features(bench_poke, active=False)
            opp_bench_count += 1
        while opp_bench_count < 5:
            numerical += self._get_pokemon_pad_numerical(active=False)
            opp_bench_count += 1

        # ── player fainted (variable-length text) ──
        fainted_str = []
        fainted_count = 0
        for fainted_poke in consistent_pokemon_order(state.fainted_pokemon):
            fainted_str += ["<fainted>"] + self._get_player_fainted_string_features(fainted_poke)
            numerical += self._get_pokemon_numerical_features(fainted_poke, active=False)
            fainted_count += 1
        while fainted_count < self.NUM_FAINTED_SLOTS:
            numerical += self._get_pokemon_pad_numerical(active=False)
            fainted_count += 1

        # ── opponent fainted (variable-length text) ──
        opp_fainted_str = []
        opp_fainted_count = 0
        for fainted_poke in consistent_pokemon_order(state.opponent_fainted):
            opp_fainted_str += ["<opponent_fainted>"] + self._get_opponent_inactive_string_features(fainted_poke)
            numerical += self._get_pokemon_numerical_features(fainted_poke, active=False)
            opp_fainted_count += 1
        while opp_fainted_count < self.NUM_OPPONENT_FAINTED_SLOTS:
            numerical += self._get_pokemon_pad_numerical(active=False)
            opp_fainted_count += 1

        global_str = ["<conditions>"] + [
            state.weather,
            state.player_conditions,
            state.opponent_conditions,
        ]
        prev_move_str = (
            ["<player_prev>"]
            + self._get_move_string_features(state.player_prev_move, active=False)
            + ["<opp_prev>"]
            + self._get_move_string_features(state.opponent_prev_move, active=False)
        )
        # terminal indicator: signals forfeit/disconnect/timer wins where
        # the losing team may still have non-fainted Pokémon on the field.
        if state.battle_won:
            terminal_token = "<won>"
        elif state.battle_lost:
            terminal_token = "<lost>"
        else:
            terminal_token = "<ongoing>"

        full_text_list = (
            [f"<{state.agent_format}>", force_switch]
            + player_str
            + move_str
            + switch_str
            + opponent_str
            + opponent_bench_str
            + fainted_str
            + opp_fainted_str
            + global_str
            + prev_move_str
            + [terminal_token]
        )
        text = " ".join(full_text_list)
        text = np.array(text, dtype=np.str_)
        numbers = np.array(numerical, dtype=np.float32)
        return {"text": text, "numbers": numbers}
