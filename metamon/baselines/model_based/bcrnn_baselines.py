import os
from functools import lru_cache
from abc import ABC, abstractmethod

import torch
from torch.distributions import Categorical

from metamon.baselines import Baseline, register_baseline, heuristic
from metamon.interface import action_idx_to_battle_order, UniversalState
from metamon.data.tokenizer import get_tokenizer
from metamon.il.model import MetamonILModel


class BCRNNBaseline(Baseline, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.load_model()
        assert isinstance(self.model, MetamonILModel)
        self.model.eval()
        self.hidden_states = {}

    @abstractmethod
    def load_model(self):
        pass

    def randomize(self):
        pass

    def battle_to_order(self, battle):
        """
        Attempt to get a valid poke-env / showdown action from the IL model

        Returns None if the model selects an invalid action.
        """
        # convert Battle to the replay parser format via UniversalState
        obs = UniversalState.from_Battle(battle).to_numpy()
        # tokenize, prepare for torch inference
        numerical = torch.from_numpy(obs["numbers"]).view(1, 1, -1)
        # the tokenizer used to train the model was also saved, protecting
        # us from any later changes to the token list.
        tokens = self.model.tokenizer.tokenize(obs["text"].tolist())
        tokens = torch.from_numpy(tokens).view(1, 1, -1)

        # grab the hidden state for this battle
        battle_id = battle.battle_tag
        if battle_id in self.hidden_states:
            hidden_state = self.hidden_states[battle_id]
        else:
            hidden_state = None

        # model inference
        with torch.inference_mode():
            raw_logits, new_hidden_state = self.model(
                token_inputs=tokens,
                numerical_inputs=numerical,
                hidden_state=hidden_state,
            )
            assert (
                raw_logits.shape[-1] <= 9
            ), "inference model appears to have too many action outputs, which will go unused"
            action_dist = Categorical(logits=raw_logits)
            action_idx = action_dist.sample().item()

        # update hidden state
        self.hidden_states[battle_id] = new_hidden_state
        order = action_idx_to_battle_order(battle, action_idx)
        return order

    def choose_move(self, battle):
        model_chosen_order = self.battle_to_order(battle)
        if model_chosen_order is not None:
            return model_chosen_order
        else:
            return self.on_invalid_il_order(battle)

    def on_invalid_il_order(self, battle):
        """
        What do we do when the model recommends an invalid move?
        """
        return self.choose_random_move(battle)


@lru_cache(maxsize=32)
def load_pretrained_model_to_cpu(model_filename):
    path = os.path.join(os.path.dirname(__file__), "pretrained_models", model_filename)
    model = torch.load(path, map_location="cpu")
    model.to("cpu")
    return model


class PretrainedOnCPU(BCRNNBaseline, ABC):
    @property
    @abstractmethod
    def model_path(self) -> str:
        pass

    def load_model(self):
        return load_pretrained_model_to_cpu(self.model_path)


@register_baseline()
class BaseRNN(PretrainedOnCPU):
    @property
    def model_path(self):
        return "replays_v2_full_trial1_BEST.pt"


@register_baseline()
class WinsOnlyRNN(PretrainedOnCPU):
    @property
    def model_path(self):
        return "replays_v2_wins_only_trial1_BEST.pt"


@register_baseline()
class MiniRNN(PretrainedOnCPU):
    @property
    def model_path(self):
        return "replays_v2_small_trial1_BEST.pt"
