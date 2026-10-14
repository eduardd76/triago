# triago

**One forward pass. Right team — or it asks.**  Local, calibrated IT ticket dispatching for GLPI.

Fixed-answer scoring (Jev-style) on Apple Silicon via MLX, wired to a GLPI demo ITSM.
No fine-tuning and no generated text: every decision is **one forward pass**, then a softmax over
the logits of the allowed label tokens (A, B, C, ...) only.

Each ticket goes through bounded decisions:

1. **Security gate** (security / normal) — calibrated; trips at p ≥ 0.05 → SOC, P1 (recall over false alarms)
2. **Assignment group** (8 groups + `other`), with the 3 most similar past tickets shown as worked examples
3. **Impact** and **urgency** (high/medium/low) → ITIL matrix in code → P1..P5
4. **Policy**: auto-assign if calibrated top ≥ 0.90 and margin ≥ 0.25, else `Service Desk Triage`

Every dispatch leaves a private followup in GLPI with the full probability breakdown.
Human reassignments are exported as labeled data and become new examples — the system learns
team conventions without training.

## Results (Qwen3-4B-Instruct-2507, 4-bit, Apple M5)

All numbers are on sets that were never used for tuning.

| test set | what it is | group accuracy | notes |
|---|---|---|---|
| Zenodo IT tickets, 657 test tickets | real MSP tickets, their own 7 queues | **80.8%**, macro-F1 **0.79** | authors' trained model: 79% / 0.72 |
| `data/real_like`, 300 tickets | realistic, agent-written, blind-reviewed | **85.0%** | 3 sources: end users / IT & alerts / 8 languages |
| `data/fresh`, 150 tickets | security stress: 40 incidents + 50 lookalikes | **83.3%** | security gate: 32/40 incidents caught, 5 false alarms |

**Known weakness:** subtle incidents (MFA push fatigue, vendor bank-detail fraud, a lost laptop mentioned in passing) are
missed by the single-question security gate about 20% of the time on the stress set. Treat the gate as a filter that
speeds up the SOC, not as a replacement for SOC alerting.

Latency: ~0.8 s per ticket for all four decisions; the question+labels prefix is KV-cached.

## Install

Requires an Apple Silicon Mac (MLX), Python ≥ 3.11 and Docker for the GLPI demo.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_zenodo.py           # optional: real-world benchmark data (CC BY 4.0)
pytest                                   # unit tests, no model download needed
```

The model (`mlx-community/Qwen3-4B-Instruct-2507-4bit`, ~2.5 GB, Apache-2.0) downloads on first run;
override with `TRIAGO_MODEL=...`.

## Run

```bash
cd itsm && docker compose up -d          # GLPI at http://localhost:8080  (glpi / glpi) — demo only
source .venv/bin/activate
python -m triago.dispatcher run             # poll GLPI every 10s and dispatch New tickets
python -m triago.dispatcher feedback        # export human reassignments -> data/feedback.jsonl

python -m triago.bench                      # fit calibration on CALIB half, report on TEST half
python -m triago.bench --eval calib         # dev loop while editing descriptions
python -m triago.bench --holdout data/fresh # fit on CALIB, report on a never-tuned set
python -m triago.bench --shots 0            # without retrieved examples
python -m triago.bench --race               # scoring vs. text generation, same question
python -m triago.bench --seed               # push data/tickets.jsonl into GLPI as New tickets
python -m triago.zenodo_bench               # real-world check on the Zenodo dataset
```

API setup used for the demo (DB, local only): `enable_api=1`, `enable_api_login_credentials=1`,
API client "full access from localhost" with IP range cleared (Docker host is not 127.0.0.1).
Default credentials and an open API client: never expose this GLPI outside your machine.

## How to change routing

Edit the label descriptions in `triago/pipeline.py` (`SECURITY`, `GROUPS`), keeping
`docs/labeling-guide.md` in sync, then run `python -m triago.bench --eval calib`. Only judge a change on
a held-out set, once. Lessons from this repo: descriptions tuned on ~30 tickets per group overfit
quickly; plain keyword lists beat nuanced sentences for a 4B model; retrieved examples help on real
queues (+7 points) because they carry team conventions.

## Files

- `triago/scorer.py` — MLX scorer: restricted softmax over label tokens, in-context single-token check, prefix KV cache, optional examples
- `triago/pipeline.py` — taxonomy, priority matrix, `Policy`, `route()` (pure, re-evaluable offline), `is_confident()`
- `triago/examples.py` — `ExampleStore`: char n-gram TF-IDF retrieval of similar labeled tickets
- `triago/glpi.py` — GLPI REST client (groups, Group_Ticket, ITILFollowup)
- `triago/dispatcher.py` — poll loop + feedback export
- `triago/bench.py` — benchmark, temperature + security-threshold fitting, calibration tables, holdout mode
- `triago/zenodo_bench.py` — real-dataset benchmark against the authors' trained baseline
- `docs/labeling-guide.md` — taxonomy and settled boundary rules used for all labels
- `data/tickets.jsonl`, `data/gen/` — 506 synthetic dev tickets (CALIB/TEST split by hash)
- `data/real_like/`, `data/fresh/` — held-out realistic sets
- `data/external/zenodo_it_tickets/` — Benitez Pereira (2022), *Classification of IT Support Tickets*, Zenodo, doi:10.5281/zenodo.7648117, CC BY 4.0

## License and credits

Code: Apache-2.0 (see `LICENSE`). Ticket sets in `data/` were written for this project with fictional
names and hosts and are released under the same license. The Zenodo dataset is not redistributed;
`scripts/fetch_zenodo.py` downloads it (CC BY 4.0, cite Benitez Pereira 2022).
The fixed-answer scoring idea follows a published write-up of Jev-style restricted-softmax scoring
over label logits (SGLang `/v1/score`); this is an independent implementation.
