import pytest

from triago import connector
from triago.pipeline import Dispatch
from triago.scorer import Decision
from triago.serve import TRIAGE_QUEUE, ticket_text, to_json


def dec(choice, p):
    return Decision(choice, {choice: p, "other": 1 - p}, 100.0)


def test_unsure_ticket_goes_to_triage_but_keeps_best_guess():
    d = Dispatch("network", 3, False, "network=0.60", {"group": dec("network", 0.6),
                                                        "impact": dec("medium", 0.8), "urgency": dec("low", 0.7)})
    out = to_json(d)
    assert out["assign_to"] == TRIAGE_QUEUE and out["group"] == "network"
    assert out["impact"] == "medium" and out["latency_ms"] == 300


def test_security_short_circuit_has_no_impact_or_urgency():
    out = to_json(Dispatch("security", 1, True, "security gate p=0.30", {"security": dec("security", 0.3)}))
    assert out["assign_to"] == "security" and out["security_incident"]
    assert out["impact"] is None and out["urgency"] is None


def test_ticket_text_accepts_subject_body_or_text():
    assert ticket_text({"subject": "VPN", "body": "drops"}) == "VPN\ndrops"
    assert ticket_text({"text": "raw"}) == "raw"
    with pytest.raises(ValueError):
        ticket_text({})


class FakeConnector:
    pass


def test_load_connector_from_spec():
    assert isinstance(connector.load(f"{__name__}:FakeConnector"), FakeConnector)
    with pytest.raises(ValueError):
        connector.load("no_colon_here")


def test_security_p_is_the_calibrated_value_the_gate_compared():
    from triago.pipeline import Policy, recalibrate
    sec = Decision("normal", {"security": 0.0, "normal": 1.0}, 50.0, {"security": -10.0, "normal": 0.0})
    out = to_json(Dispatch("network", 3, True, "", {"security": sec}), Policy())
    assert out["security_p"] == round(recalibrate(sec, Policy().security_temperature).probabilities["security"], 4)
    assert out["security_trip"] == Policy().security_trip


def test_ui_page_ships_with_the_package():
    from triago.serve import PAGE
    assert PAGE.exists() and b"/dispatch" in PAGE.read_bytes()
