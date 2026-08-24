from jarvis_nlu.intents import (
    ALL_INTENTS, CHORE_INTENTS, DESTRUCTIVE_INTENTS, Intent, SMALLTALK_INTENTS,
)


def test_v1_has_exactly_twenty_intents():
    assert len(ALL_INTENTS) == 20


def test_intent_values_are_unique():
    assert len({i.value for i in ALL_INTENTS}) == 20


def test_out_of_scope_is_not_a_chore_or_smalltalk():
    assert Intent.OUT_OF_SCOPE not in CHORE_INTENTS
    assert Intent.OUT_OF_SCOPE not in SMALLTALK_INTENTS


def test_cancel_reminder_is_destructive():
    assert Intent.CANCEL_REMINDER in DESTRUCTIVE_INTENTS


def test_groups_are_subsets_of_all_intents():
    for group in (CHORE_INTENTS, SMALLTALK_INTENTS, DESTRUCTIVE_INTENTS):
        assert group <= set(ALL_INTENTS)
