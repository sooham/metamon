import pytest

from metamon.jepa.checkpointing import (
    require_tokenizer_from_checkpoint,
    save_paired_jepa_checkpoint,
)
from metamon.tokenizer import PokemonTokenizer


class DummyModel:
    def __init__(self):
        self.saved_path = None
        self.saved_extra = None

    def save_checkpoint(self, path: str, **extra) -> None:
        self.saved_path = path
        self.saved_extra = extra


def test_save_paired_jepa_checkpoint_includes_inference_metadata():
    tokenizer = PokemonTokenizer()
    tokenizer.name = "unit-tokenizer"
    model = DummyModel()

    save_paired_jepa_checkpoint(
        model,
        "/tmp/paired.pt",
        epoch=2,
        global_step=17,
        config={"latent_dim": 8},
        vocab_size=123,
        max_history_blocks=4,
        tokenizer=tokenizer,
        best_val_loss=0.123,
        best_val_epoch=2,
        best_val_global_step=17,
        best_val_metrics={"val_loss": 0.123, "val_next_state_loss": 0.04},
        last_val_metrics={"val_loss": 0.125},
    )

    assert model.saved_path == "/tmp/paired.pt"
    assert model.saved_extra["epoch"] == 2
    assert model.saved_extra["global_step"] == 17
    assert model.saved_extra["config"] == {"latent_dim": 8}
    assert model.saved_extra["vocab_size"] == 123
    assert model.saved_extra["max_history_blocks"] == 4
    assert model.saved_extra["best_val_loss"] == 0.123
    assert model.saved_extra["best_val_epoch"] == 2
    assert model.saved_extra["best_val_global_step"] == 17
    assert model.saved_extra["best_val_metrics"] == {
        "val_loss": 0.123,
        "val_next_state_loss": 0.04,
    }
    assert model.saved_extra["last_val_metrics"] == {"val_loss": 0.125}
    restored = require_tokenizer_from_checkpoint(
        model.saved_extra,
        "/tmp/paired.pt",
    )
    assert restored.name == "unit-tokenizer"


def test_require_tokenizer_from_checkpoint_rejects_missing_state():
    with pytest.raises(ValueError, match="tokenizer_state"):
        require_tokenizer_from_checkpoint({}, "/tmp/old.pt")
