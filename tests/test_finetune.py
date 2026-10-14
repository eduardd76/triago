from triago.finetune import Task, encode_examples, scrub
from triago.scorer import Scorer

from test_scorer import CharTokenizer


def test_scrub_removes_contact_details_and_secrets():
    t = scrub("mail jan.smit@acme.nl or +31 6 1234 5678, host 10.1.2.3, password: Hunter2! see https://x.io/a")
    assert "jan.smit" not in t and "1234" not in t and "10.1.2.3" not in t and "Hunter2" not in t and "x.io" not in t
    assert "<EMAIL>" in t and "<PHONE>" in t and "<IP>" in t and "<SECRET>" in t and "<URL>" in t


def test_scrub_keeps_ordinary_ticket_text():
    t = "VPN drops every 10 minutes since Monday, ticket INC-4411, Outlook 2016"
    assert scrub(t) == t


def test_training_tokens_are_the_scoring_prompt_plus_the_answer_letter():
    s = Scorer.__new__(Scorer)
    s.tokenizer, s._label_ids_cache = CharTokenizer(), {}
    task = Task("group", "Which team?", {"network": "Network", "other": "Other"}, [], [], [])
    (tokens, offset), = encode_examples(s, task, [("VPN drops", "other")], None, 0)
    prompt, _ = s._render("VPN drops", "Which team?", task.choices)
    assert tokens[:offset] == s._encode(prompt) and tokens[offset:] == [ord("B")]
