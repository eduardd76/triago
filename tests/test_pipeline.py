import itertools

import pytest

from triago.pipeline import GROUPS, IMPACT, PRIORITY_MATRIX, URGENCY, Policy, is_confident, recalibrate, route
from triago.scorer import Decision


def dec(logits: dict[str, float]) -> Decision:
    return recalibrate(Decision(None, {}, 1.0, logits), 1.0)


def raw(sec=-40.0, group="network", impact="low", urgency="low"):
    g = {k: (8.0 if k == group else 0.0) for k in GROUPS}
    return {
        "security": dec({"security": sec, "normal": 5.0}),
        "group": dec(g),
        "impact": dec({k: (5.0 if k == impact else 0.0) for k in IMPACT}),
        "urgency": dec({k: (5.0 if k == urgency else 0.0) for k in URGENCY}),
    }


def test_recalibrate_keeps_argmax_and_softens():
    d = dec({"a": 4.0, "b": 1.0, "c": 0.0})
    soft = recalibrate(d, 5.0)
    assert soft.choice == d.choice == "a"
    assert soft.probabilities["a"] < d.probabilities["a"]
    assert sum(soft.probabilities.values()) == pytest.approx(1.0)


def test_priority_matrix_is_complete_and_itil_shaped():
    assert set(PRIORITY_MATRIX) == set(itertools.product(IMPACT, URGENCY))
    assert PRIORITY_MATRIX[("high", "high")] == 1
    assert PRIORITY_MATRIX[("low", "low")] == 5


def test_security_gate_short_circuits_to_soc_p1():
    d = route(raw(sec=9.0), Policy())
    assert (d.group, d.priority, d.auto) == ("security", 1, True)


def test_normal_ticket_gets_group_and_matrix_priority():
    d = route(raw(group="telephony", impact="high", urgency="medium"), Policy(temperature=1.0))
    assert (d.group, d.priority, d.auto) == ("telephony", 2, True)


def test_other_is_never_auto_dispatched():
    d = route(raw(group="other"), Policy(temperature=1.0))
    assert d.group == "other" and not d.auto


def test_low_confidence_goes_to_triage():
    flat = dec({k: 0.0 for k in GROUPS})
    assert not is_confident(flat, Policy())


def test_security_gate_trips_on_weak_signal():
    # A 10-logit lead for "normal" is still p~0.15 after calibration (T=5.75) -> above the 0.05 trip.
    assert route(raw(sec=-5.0), Policy()).group == "security"
