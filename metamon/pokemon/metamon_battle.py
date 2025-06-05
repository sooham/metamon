import os
import re
from logging import Logger
from typing import Any, Dict, List, Optional, Set, Tuple, Union


import poke_env.environment as pe


class MetamonBackendBattle(pe.AbstractBattle):

    def __init__(
        self,
        battle_tag: str,
        username: str,
        logger: Logger,
        save_replays: Union[str, bool],
        gen: int,
    ):
        self._battle_tag = battle_tag
        self._username = username
        self._logger = logger
        self._save_replays = save_replays

    def _finish_battle(self):
        super()._finish_battle()

    def parse_message(self, split_message: List[str]):
        raise NotImplementedError

    def parse_request(self, request: Dict[str, Any]):
        raise NotImplementedError

    @property
    def active_pokemon(self) -> Any:
        raise NotImplementedError

    @property
    def all_active_pokemons(self) -> List[Optional[pe.Pokemon]]:
        raise NotImplementedError

    @property
    def available_moves(self) -> Any:
        raise NotImplementedError

    @property
    def available_switches(self) -> Any:
        raise NotImplementedError

    @property
    def battle_tag(self) -> str:
        """
        :return: The battle identifier.
        :rtype: str
        """
        raise NotImplementedError

    def clear_all_boosts(self):
        pass

    @property
    def dynamax_turns_left(self) -> Optional[int]:
        """
        :return: How many turns of dynamax are left. None if dynamax is not active
        :rtype: int, optional
        """
        return None

    def end_illusion(self):
        pass

    @property
    def fields(self) -> Dict[pe.Field, int]:
        """
        :return: A Dict mapping fields to the turn they have been activated.
        :rtype: Dict[Field, int]
        """
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        """
        :return: A boolean indicating whether the battle is finished.
        :rtype: Optional[bool]
        """
        raise NotImplementedError

    @property
    def force_switch(self) -> Any:
        raise NotImplementedError

    @property
    def format(self) -> Optional[str]:
        """
        :return: The format of the battle, in accordance with Showdown protocol
        :rtype: Optional[str]
        """
        raise NotImplementedError

    @property
    def gen(self) -> int:
        """
        :return: The generation of the battle; will be the parameter with which the
            the battle was initiated
        :rtype: int
        """
        raise NotImplementedError

    @property
    def last_request(self) -> Dict[str, Any]:
        """
        The last request received from the server. This allows players to track
            rqid and also maintain parallel battle copies for search/inference

        :return: The last request.
        :rtype: Dict[str, Any]
        """
        return self._last_request

    @property
    def lost(self) -> Optional[bool]:
        """
        :return: If the battle is finished, a boolean indicating whether the battle is
            lost. Otherwise None.
        :rtype: Optional[bool]
        """
        return None if self._won is None else not self._won

    @property
    def max_team_size(self) -> Optional[int]:
        """
        :return: The maximum acceptable size of the team to return in teampreview, if
            applicable.
        :rtype: int, optional
        """
        return self._max_team_size

    @property
    def maybe_trapped(self) -> Any:
        raise NotImplementedError

    @property
    def opponent_active_pokemon(self) -> Any:
        raise NotImplementedError

    @property
    def opponent_can_dynamax(self) -> Any:
        raise NotImplementedError

    @opponent_can_dynamax.setter
    def opponent_can_dynamax(self, value: bool) -> Any:
        raise NotImplementedError

    @property
    def opponent_dynamax_turns_left(self) -> Optional[int]:
        """
        :return: How many turns of dynamax are left for the opponent's pokemon.
            None if dynamax is not active
        :rtype: int | None
        """
        return None

    @property
    def opponent_role(self) -> Optional[str]:
        """
        :return: Opponent's role in given battle. p1/p2
        :rtype: str, optional
        """
        if self.player_role == "p1":
            return "p2"
        if self.player_role == "p2":
            return "p1"
        return None

    @property
    def opponent_side_conditions(self) -> Dict[pe.SideCondition, int]:
        """
        :return: The opponent's side conditions. Keys are SideCondition objects, values
            are:

            - the number of layers of the SideCondition if the side condition is
                stackable
            - the turn where the SideCondition was setup otherwise
        :rtype: Dict[SideCondition, int]
        """
        raise NotImplementedError

    @property
    def opponent_team(self) -> Dict[str, pe.Pokemon]:
        """
        During teampreview, keys are not definitive: please rely on values.

        :return: The opponent's team. Keys are identifiers, values are pokemon objects.
        :rtype: Dict[str, Pokemon]
        """
        raise NotImplementedError

    @property
    def opponent_username(self) -> Optional[str]:
        """
        :return: The opponent's username, or None if unknown.
        :rtype: str, optional.
        """
        return self._opponent_username

    @opponent_username.setter
    def opponent_username(self, value: str):
        self._opponent_username = value

    @property
    def player_role(self) -> Optional[str]:
        """
        :return: Player's role in given battle. p1/p2
        :rtype: str, optional
        """
        return self._player_role

    @player_role.setter
    def player_role(self, value: Optional[str]):
        self._player_role = value

    @property
    def player_username(self) -> str:
        """
        :return: The player's username.
        :rtype: str
        """
        raise NotImplementedError

    @property
    def players(self) -> Tuple[str, str]:
        """
        :return: The pair of players' usernames.
        :rtype: Tuple[str, str]
        """
        return self._players

    @players.setter
    def players(self, players: Tuple[str, str]):
        """Sets the battle player's name:

        :param player_1: First player's username.
        :type player_1: str
        :param player_1: Second player's username.
        :type player_2: str
        """
        self._players = players

    @property
    def rating(self) -> Optional[int]:
        """
        Player's rating after the end of the battle, if it was received.

        :return: The player's rating after the end of the battle.
        :rtype: int, optional
        """
        raise NotImplementedError

    @property
    def opponent_rating(self) -> Optional[int]:
        """
        Opponent's rating after the end of the battle, if it was received.

        :return: The opponent's rating after the end of the battle.
        :rtype: int, optional
        """
        raise NotImplementedError

    @property
    def side_conditions(self) -> Dict[pe.SideCondition, int]:
        """
        :return: The player's side conditions. Keys are SideCondition objects, values
            are:

            - the number of layers of the side condition if the side condition is
                stackable
            - the turn where the SideCondition was setup otherwise
        :rtype: Dict[SideCondition, int]
        """
        raise NotImplementedError

    def switch(self, pokemon_str: str, details: str, hp_status: str):
        pass

    @property
    def team(self) -> Dict[str, pe.Pokemon]:
        """
        :return: The player's team. Keys are identifiers, values are pokemon objects.
        :rtype: Dict[str, Pokemon]
        """
        return self._team

    @team.setter
    def team(self, value: Dict[str, pe.Pokemon]):
        self._team = value

    @property
    def team_size(self) -> int:
        """
        :return: The number of Pokemon in the player's team.
        :rtype: int
        """
        raise NotImplementedError

    @property
    def trapped(self) -> Any:
        raise NotImplementedError

    @trapped.setter
    def trapped(self, value: Any):
        raise NotImplementedError

    @property
    def turn(self) -> int:
        """
        :return: The current battle turn.
        :rtype: int
        """
        raise NotImplementedError

    @property
    def weather(self) -> Dict[pe.Weather, int]:
        """
        :return: A Dict mapping the battle's weather (if any) to its starting turn
        :rtype: Dict[Weather, int]
        """
        raise NotImplementedError

    @property
    def won(self) -> Optional[bool]:
        """
        :return: If the battle is finished, a boolean indicating whether the battle is
            won. Otherwise None.
        :rtype: Optional[bool]
        """
        raise NotImplementedError

    @property
    def move_on_next_request(self) -> bool:
        """
        :return: Wheter the next received request should yield a move order directly.
            This can happen when a switch is forced, or an error is encountered.
        :rtype: bool
        """
        return self._move_on_next_request

    @move_on_next_request.setter
    def move_on_next_request(self, value: bool):
        self._move_on_next_request = value

    @property
    def reviving(self) -> bool:
        raise NotImplementedError

    @property
    def maybe_trapped(self) -> Any:
        raise NotImplementedError

    @property
    def can_dynamax(self) -> bool:
        raise NotImplementedError

    @property
    def can_mega_evolve(self) -> bool:
        raise NotImplementedError

    @property
    def can_z_move(self) -> bool:
        raise NotImplementedError

    @property
    def can_tera(self) -> bool:
        raise NotImplementedError

    @property
    def opponent_active_pokemon(self) -> Any:
        raise NotImplementedError

    @property
    def opponent_can_dynamax(self) -> bool:
        raise NotImplementedError

    # @opponent_can_dynamax.setter
    # def opponent_can_dynamax(self, value: bool) -> Any:
    #     self._opponent_can_dynamax = value
