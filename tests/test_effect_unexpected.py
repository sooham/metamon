"""Test that Showdown effects TWOTURNMOVE and PARSPEEDDROP resolve correctly."""

import pytest
from metamon.backend.replay_parser.pe_datatypes import PEEffect


def test_twoturnmove_effect_resolves():
    """TWOTURNMOVE is used for Fly, Dig, Solar Beam, etc. charging turns."""
    effect = PEEffect.from_showdown_message("twoturnmove")
    assert effect == PEEffect.TWOTURNMOVE
    assert effect != PEEffect.UNKNOWN


def test_parspeeddrop_effect_resolves():
    """PARSPEEDDROP is the Speed reduction from paralysis (Gen 8+ mechanic)."""
    # Showdown emits this as "Par Speed Drop"
    effect = PEEffect.from_showdown_message("Par Speed Drop")
    assert effect == PEEffect.PARSPEEDDROP
    assert effect != PEEffect.UNKNOWN
