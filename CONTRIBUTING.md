# Contributing to triago

Thanks for helping. Small, focused pull requests are easiest to review.

## Set up

```bash
git clone https://github.com/eduardd76/triago && cd triago
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # Apple Silicon; use ".[torch,dev]" elsewhere
pytest                             # ~25 tests, a few seconds, no model download
```

CI runs the same tests on macOS, Ubuntu and Windows for every push and pull request.

## Good first contributions

- **A connector for your helpdesk** (Zammad, osTicket, Jira SM, ...): two methods, see
  [docs/integration.md](docs/integration.md#python-connector) and `triago/glpi.py` as the reference.
- **PyTorch fine-tuning**, so Linux/Windows users can train adapters (mirror `triago/finetune.py`).
- **Speed numbers on NVIDIA GPUs** or CPUs, measured with `python -m triago.bench`.
- **Labeled tickets in more languages** (fictional names and hosts only).

## Rules of thumb

- **Routing changes need numbers.** If you change team descriptions, prompts or thresholds, include
  `python -m triago.bench` output before and after, and judge on a held-out set
  (`--holdout data/real_like` or `data/fresh`), not on the tickets you tuned against.
- **No real tickets in the repo.** Data in `data/` must be fictional. Never commit `data/feedback.jsonl`,
  `adapters/` or anything exported from a real helpdesk (they are git-ignored).
- **Keep it local.** triago must work fully offline after the model download: no telemetry, no external calls.
- **Tests:** new logic gets a unit test that runs without downloading a model (see `tests/test_scorer.py`
  for the fake-tokenizer pattern).

## Reporting a security issue

Please don't open a public issue for vulnerabilities. Use GitHub's
[private vulnerability reporting](https://github.com/eduardd76/triago/security/advisories/new).
