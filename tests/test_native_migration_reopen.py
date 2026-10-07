"""Deterministic coverage for migration reopen sampling."""
from __future__ import annotations

import pytest

from scripts.verify_native_migration_reopen import sampled_actions


def test_reopen_first_middle_last_action() -> None:
    assert sampled_actions([{"action_index": index} for index in range(6)]) == (0, 3, 5)
    assert sampled_actions([{"action_index": 0}]) == (0,)


def test_reopen_rejects_missing_action() -> None:
    with pytest.raises(ValueError, match="noncontiguous"):
        sampled_actions([{"action_index": 0}, {"action_index": 2}])
