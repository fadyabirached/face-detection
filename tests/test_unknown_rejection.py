"""
Tests for `decide_identity` — the margin-based open-set rejection logic
in app.py. Pure function, no model/dataset needed, but `app.py` loads
the model at import time, so every test goes through the `app_module`
fixture (see conftest.py) rather than a bare `import app`.
"""


def test_decisive_top1_is_accepted(app_module):
    # Clear winner: 70% vs 20% vs 10% -> margin ratio 3.5
    name, conf, is_known = app_module.decide_identity(
        ["alice", "bob", "carol"], [70.0, 20.0, 10.0]
    )
    assert (name, conf, is_known) == ("alice", 70.0, True)


def test_low_confidence_but_decisive_margin_is_still_accepted(app_module):
    """The "Emma Watson" case: an identity with few training images can
    legitimately have a low absolute top-1 confidence while still being
    clearly ahead of every other candidate. Rejecting on raw confidence
    alone would wrongly call this person Unknown."""
    name, conf, is_known = app_module.decide_identity(
        ["emma_watson", "someone_else", "another_person"], [22.0, 8.0, 7.0]
    )
    assert name == "emma_watson"
    assert conf == 22.0
    assert is_known is True


def test_close_race_between_top_candidates_is_rejected(app_module):
    """A true unknown face tends to spread weak, similar confidence
    across several candidates with no clear winner."""
    name, conf, is_known = app_module.decide_identity(
        ["alice", "bob", "carol"], [12.0, 11.0, 10.0]
    )
    assert is_known is False
    assert name == "Unknown"


def test_near_uniform_noise_is_rejected_even_with_a_technical_margin(app_module):
    """Guards against the degenerate case: everything is near-zero, so a
    tiny margin ratio could exist by chance without meaning anything."""
    name, conf, is_known = app_module.decide_identity(
        ["alice", "bob", "carol"], [2.5, 1.0, 0.9]
    )
    assert is_known is False


def test_no_candidates_returns_unknown(app_module):
    name, conf, is_known = app_module.decide_identity([], [])
    assert (name, conf, is_known) == ("Unknown", 0.0, False)


def test_single_candidate_with_no_runner_up_uses_absolute_floor_only(app_module):
    # No runner-up to compare against -> margin ratio is infinite, so
    # this only needs to clear the absolute confidence floor.
    name, conf, is_known = app_module.decide_identity(["alice"], [50.0])
    assert is_known is True

    name, conf, is_known = app_module.decide_identity(["alice"], [1.0])
    assert is_known is False


def test_custom_thresholds_are_respected(app_module):
    # Would be rejected under defaults (margin ratio 1.2 < 1.5), but
    # accepted with a looser custom threshold.
    name, conf, is_known = app_module.decide_identity(
        ["alice", "bob"], [24.0, 20.0], min_margin_ratio=1.1
    )
    assert is_known is True
