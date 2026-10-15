# Make triago route like your team

Four levers, cheapest first. Most teams only need the first three.

## 1. Your teams

The teams and their descriptions live in `GROUPS` in `triago/pipeline.py`:

```python
GROUPS = {
    "network": ("Network: wifi/LAN/WAN for a site or many users, VPN connectivity, firewall rules ..."),
    "server_infra": ("Server & Infrastructure: Windows/Linux servers, VMs, storage, file shares ..."),
    ...
    "other": ("NOT an IT matter or no actionable content: HR, payroll, ..."),
}
```

- Rename, add or remove teams. Keep `other` (the "not IT" escape hatch). At most 26, one letter each.
- Write descriptions as **keyword lists of what the team owns**, not as nuanced sentences. For a 4B model,
  plain keywords work better.
- Keep the security gate's description (`SECURITY`) focused on actual incidents.
- GLPI: if you change the keys, update the mapping in `triago/glpi.py` (`GROUP_NAMES`) and
  `integrations/glpi/triago/hook.php` (`PLUGIN_TRIAGO_GROUPS`).
- Decide your boundary rules ("shared drives → Server & Infrastructure") and write them down;
  `docs/labeling-guide.md` is the one used for the demo data.

## 2. Your history: past tickets as examples

For every ticket, triago shows the model the 3 most similar labeled tickets it knows. This is the biggest,
cheapest win: **+7 points on real tickets**, no training.

Export a few hundred resolved tickets with the team that actually handled them, one per line, to
`data/gen/yours.jsonl`:

```json
{"t": "VPN drops every 10 minutes | Since Monday the VPN disconnects for the whole Lyon office.", "g": "network"}
{"t": "Need access to the Finance share | Please give me read access to \\\\fs01\\finance.", "g": "identity_access"}
```

`t` is `"<subject> | <body>"`, `g` is a key from `GROUPS`. The server and dispatcher load every
`data/gen/*.jsonl` file, and reload when one changes.

## 3. Your corrections: the nightly feedback loop

Whenever a technician moves a ticket to another team, that's a lesson. Export them:

```bash
python -m triago.dispatcher feedback          # GLPI -> data/feedback.jsonl
scripts/schedule_feedback_macos.sh            # same thing every night at 02:00 (--remove to stop)
```

The export rewrites `data/feedback.jsonl` from GLPI each time, so GLPI is the source of truth: to undo a
wrong correction, fix the ticket's team in GLPI. Corrections win over older examples with the same text.
For other helpdesks, write the same file yourself: one `{"text": ..., "group": ...}` per line.

## 4. Calibration: so that "90% sure" stays true

Each question has a *temperature* that turns raw model scores into honest probabilities. The defaults in
`Policy` (`triago/pipeline.py`) were fitted for the demo teams and the MLX 4-bit model. Re-fit them when
you change the teams, the model or the backend, or once you have your own labeled tickets:

```bash
python -m triago.bench                       # fits on half the labeled tickets, reports on the other half
```

It prints the fitted `T` (team question) and security temperature, the security trip point, and a table
of "threshold → share of tickets auto-assigned → accuracy". Put the values into `Policy` and, if you like,
move `min_top` (default 0.90) to trade fewer human touches for more mistakes.

Useful options: `--eval calib` while editing descriptions (fast loop), `--holdout data/real_like` to report
on a set never used for tuning, `--shots 0` to measure without examples, `--race` to compare scoring
with ordinary text generation.

**Judge changes on held-out tickets, once.** Descriptions tuned against a few dozen tickets per team
overfit quickly: they look better on the tickets you tuned on and worse on new ones.

## 5. Optional: fine-tuning

Once you have a few hundred to a few thousand tickets with trusted teams (your examples + corrections),
you can train a small adapter on your Mac. It took 40 minutes for 868 tickets on an M5.

```bash
python -m triago.finetune data/my_tickets.jsonl   # one {"t": ..., "g": ...} per line
export TRIAGO_ADAPTER=adapters/my_tickets         # only if it prints ACCEPTED; undo: unset TRIAGO_ADAPTER
```

What it does, so you can trust it:

1. **Scrubs** e-mails, URLs, IP and MAC addresses, phone numbers and passwords from the text
   (it does **not** detect people's names).
2. **Splits** tickets into train / valid / test by a hash of the text; test is never trained or tuned on.
3. **Trains** a LoRA adapter (~30 MB) on exactly the prompt the scorer uses, with the loss on the answer letter only.
4. **Re-fits calibration** on valid, and stores it with the adapter.
5. **Accepts the adapter only if** it beats the base model on valid and does not miss more security incidents.

Measured on 657 real tickets (Zenodo), with "unseen" = test tickets whose text never appeared in training:

| setup | test | unseen |
|---|---|---|
| base model, no examples | 73.4% | 69.9% |
| base model + 3 examples (the default) | 79.9% | 77.3% |
| fine-tuned, no examples | 83.1% | 80.1% |
| **fine-tuned + 3 examples** | **88.0%** | **85.8%** |

Nothing leaves your machine, but the adapter has *seen* your tickets: keep it as private as the tickets
(`adapters/` is git-ignored). Training needs an Apple Silicon Mac for now; the PyTorch backend can load
PEFT adapters, but a PyTorch training script is not included yet.

Reproduce the table with `python scripts/fetch_zenodo.py && python -m triago.finetune --zenodo`.
