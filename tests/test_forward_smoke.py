"""
Smoke tests for forward parsing.

Verify that ``forward_fill`` does not crash on a representative sample of
raw replays across all supported generations.
"""

import pytest
from metamon.backend.replay_parser.exceptions import ForwardException

from tests.helpers import find_random_replay_files, run_forward_fill, SUPPORTED_GENS


class TestForwardSmoke:
    """Parse a handful of replays per gen and ensure most succeed."""

    @pytest.mark.parametrize("fmt", list(SUPPORTED_GENS.keys()))
    def test_parse_N_replays(self, fmt):
        """Parse up to 10 replays from *fmt*.

        Some may fail with expected ``ForwardException`` types (Scalemons,
        NoSpeciesClause, UnfinishedReplay, etc.).  The test only fails if
        *every* replay crashes.
        """
        paths = find_random_replay_files(fmt, 10)
        assert paths, f"No raw replays found for {fmt}"

        ok, skipped = 0, 0
        for p in paths:
            try:
                replay = run_forward_fill(p)
                assert replay.gen is not None
                assert len(replay.turnlist) >= 1
                ok += 1
            except ForwardException:
                skipped += 1

        assert ok > 0, (
            f"All {len(paths)} replays for {fmt} raised ForwardException. "
            f"This may indicate a systemic parsing bug in forward_fill."
        )
        print(f"  {fmt}: {ok} OK, {skipped} skipped (expected failures)")

    def test_at_least_one_replay_per_gen(self):
        """Sanity check: every supported gen has raw replays on disk."""
        for fmt in SUPPORTED_GENS:
            paths = find_random_replay_files(fmt, 1)
            assert paths, f"No replays found for {fmt} — check $METAMON_CACHE_DIR"
