import pytest

from triago.scorer import Scorer


class CharTokenizer:
    """One token per character; optionally splits uppercase letters into two tokens."""

    def __init__(self, split_upper=False):
        self.split_upper = split_upper

    def encode(self, text, add_special_tokens=False):
        ids = []
        for c in text:
            ids += [ord(c), 0] if (self.split_upper and c.isupper()) else [ord(c)]
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


def scorer(tok) -> Scorer:
    s = Scorer.__new__(Scorer)  # no model load
    s.tokenizer, s._label_ids_cache = tok, {}
    return s


CHOICES = {"network": "Network", "end_user": "End-user", "other": "Other"}


def test_render_lists_labels_and_examples_with_letters():
    s = scorer(CharTokenizer())
    prompt, labels = s._render("wifi down", "Which group?", CHOICES, [("vpn broken", "network"), ("x", "nope")])
    assert labels == ["A", "B", "C"]
    assert "A = Network" in prompt and "- vpn broken -> A" in prompt
    assert "nope" not in prompt  # examples with unknown groups are dropped
    assert prompt.index("Allowed labels") < prompt.index("Similar past tickets") < prompt.index("Ticket:\nwifi down")


def test_label_tokens_resolved_in_context():
    s = scorer(CharTokenizer())
    prompt, labels = s._render("t", "q?", CHOICES)
    assert s._label_token_ids(prompt, labels) == [ord("A"), ord("B"), ord("C")]


def test_multi_token_label_is_rejected():
    s = scorer(CharTokenizer(split_upper=True))
    prompt, labels = s._render("t", "q?", CHOICES)
    with pytest.raises(ValueError, match="single token"):
        s._label_token_ids(prompt, labels)
