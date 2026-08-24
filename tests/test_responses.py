import pytest

from jarvis_nlu.intents import SMALLTALK_INTENTS
from jarvis_nlu.responses import POOLS, ResponsePool


def test_every_smalltalk_intent_has_a_pool():
    for intent in SMALLTALK_INTENTS:
        assert intent.value in POOLS, f"missing pool for {intent.value}"


def test_pools_meet_minimum_variant_count():
    for key, variants in POOLS.items():
        assert len(variants) >= 8, f"{key} has only {len(variants)} variants"


def test_pool_variants_are_unique():
    for key, variants in POOLS.items():
        assert len(set(variants)) == len(variants), f"{key} has duplicates"


def test_pick_avoids_recent_repeats():
    pool = ResponsePool({"greeting": ["a", "b", "c", "d"]}, history=3)
    seen = [pool.pick("greeting") for _ in range(4)]
    # With history=3, the first four picks cannot repeat.
    assert len(set(seen)) == 4


def test_pick_recycles_once_history_rolls_over():
    pool = ResponsePool({"greeting": ["a", "b", "c", "d"]}, history=3)
    picks = [pool.pick("greeting") for _ in range(20)]
    assert set(picks) == {"a", "b", "c", "d"}


def test_pick_handles_pool_smaller_than_history():
    pool = ResponsePool({"thanks": ["only"]}, history=3)
    assert pool.pick("thanks") == "only"
    assert pool.pick("thanks") == "only"


def test_pick_raises_for_unknown_key():
    with pytest.raises(KeyError):
        ResponsePool({"greeting": ["a"]}).pick("nope")


def test_history_is_per_key():
    pool = ResponsePool({"a": ["x", "y"], "b": ["x", "y"]}, history=1)
    first = pool.pick("a")
    assert pool.pick("b") in {"x", "y"}  # b's history is independent
    assert pool.pick("a") != first
