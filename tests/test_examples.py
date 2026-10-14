from triago.examples import ExampleStore

ROWS = [
    ("VPN drops every 10 minutes | whole sales team at home", "network"),
    ("Printer jams on floor 2 | the HP printer near reception", "end_user"),
    ("Reset my password | locked out after holiday", "identity_access"),
    ("Phishing email clicked | entered my password on a fake page", "security"),
]


def test_nearest_finds_lexically_similar_ticket():
    store = ExampleStore(ROWS)
    (text, group), *_ = store.nearest("vpn keeps dropping for the team", k=1)
    assert group == "network"


def test_ticket_is_never_its_own_example():
    store = ExampleStore(ROWS)
    assert all(t != ROWS[0][0] for t, _ in store.nearest(ROWS[0][0], k=3))


def test_allowed_filters_groups_and_k_zero_is_empty():
    store = ExampleStore(ROWS)
    assert all(g != "security" for _, g in store.nearest("password", k=4, allowed={"identity_access", "network", "end_user"}))
    assert store.nearest("anything", k=0) == []


def test_later_duplicate_wins():
    store = ExampleStore([("same text", "network"), ("Same  text", "application")])
    assert store.rows == [("Same  text", "application")]
