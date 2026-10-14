"""PyTorch backend on a tiny random Qwen3 (a few MB): checks the plumbing, not the routing quality."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from triago.scorer import Scorer  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
CHOICES = {"network": "Network", "end_user": "End-user", "other": "Other"}


@pytest.fixture(scope="module")
def scorer():
    try:
        return Scorer(model_path=TINY, adapter_path=None, backend="torch")
    except OSError as e:  # offline
        pytest.skip(f"tiny model not available: {e}")


def test_prefix_cache_gives_the_same_logits_as_a_full_pass(scorer):
    prompt, labels = scorer._render("VPN drops every 10 minutes", "Which team?", CHOICES)
    ids = scorer._encode(prompt)
    full = scorer.backend.next_logits(ids)
    cache = scorer.backend.prefill(ids[:-7])
    cached = scorer.backend.next_logits(ids[-7:], cache)
    assert abs(full - cached).max() < 1e-3
    again = scorer.backend.next_logits(ids[-7:], cache)  # cache was rewound to the prefix
    assert abs(cached - again).max() < 1e-5


def test_decide_returns_a_distribution_over_the_choices(scorer):
    d1 = scorer.decide("printer jam on floor 3", "Which team?", CHOICES)
    d2 = scorer.decide("printer jam on floor 3", "Which team?", CHOICES)
    assert set(d1.probabilities) == set(CHOICES)
    assert abs(sum(d1.probabilities.values()) - 1) < 1e-6
    assert d1.probabilities == pytest.approx(d2.probabilities, abs=1e-4)
