# Connecting triago to your ticketing system

There are three ways in. Pick the first one that fits:

| | your system… | triago… | ticket opens routed? |
|---|---|---|---|
| [GLPI plugin](#glpi) | is GLPI | is called while GLPI saves the ticket | yes |
| [HTTP API](#http-api) | can call a URL when a ticket is created (webhook, automation rule, script) | answers with team + priority; your system applies it | yes, if your rule runs on create |
| [Python connector](#python-connector) | has an API, but cannot call out | polls it, applies the result, writes a note | after the next poll (10 s) |

In all three, the model runs on your machine and nothing is sent anywhere else.

## GLPI

Everything is in the repo: a demo GLPI, a one-time setup script, a plugin and a poller.

```bash
cd itsm && docker compose up -d && cd ..     # GLPI 12 at http://localhost:8080 (glpi / glpi)
scripts/setup_glpi.sh                        # enables the API, installs the plugin, creates the 9 groups
python -m triago.serve                       # the plugin calls this at host.docker.internal:8642
python -m triago.dispatcher run              # safety net (see below)
```

**What the plugin does** (`integrations/glpi/triago/`, two PHP hooks):

1. Before GLPI saves a new ticket, it sends the title and description to `POST /dispatch`.
2. It adds the team (a GLPI group), impact, urgency and priority to the ticket being saved.
3. After saving, it adds triago's explanation as a **private** follow-up (technicians only).

If triago is down or slower than 4 seconds, the ticket is saved normally, unrouted, and the **dispatcher**
routes it on its next pass (every 10 s). It also routes tickets created by other channels (mail collector,
API) and re-routes a ticket whose team was removed.

**Your own GLPI** (not the demo): copy `integrations/glpi/triago/` to GLPI's `plugins/` folder, install and
activate it (*Setup → Plugins*, or `php bin/console plugin:install triago -u <admin>` and
`plugin:activate triago`), and set `TRIAGO_URL` in the web server's environment if triago is not at
`http://host.docker.internal:8642` (only the default URL has been tested so far). Create groups named like the ones in `integrations/glpi/triago/hook.php`
(or edit that mapping to your group names). The dispatcher and feedback export also need the REST API
enabled (*Setup → General → API*) and `GLPI_URL`, `GLPI_USER`, `GLPI_PASSWORD`.

**Teaching it** (see the [learning loop](how-it-works.md#the-learning-loop)): technicians reassign tickets as
usual. `python -m triago.dispatcher feedback` exports every routed ticket with its final team to
`data/feedback.jsonl`; `scripts/schedule_feedback_macos.sh` runs it nightly (a cron line for Linux is inside).
The server and dispatcher pick up the new file on the next ticket, no restart.

## HTTP API

Start the server, then call it from your helpdesk's "ticket created" automation:

```bash
python -m triago.serve                                          # 127.0.0.1:8642, no key
TRIAGO_API_KEY=change-me TRIAGO_HOST=0.0.0.0 python -m triago.serve   # reachable from other machines
```

```bash
curl -s localhost:8642/dispatch -H "Authorization: Bearer change-me" \
     -d '{"subject": "VPN drops every 10 min", "body": "Since Monday, the whole Lyon office."}'
```

Send `{"subject": ..., "body": ...}` (plain text) or `{"text": ...}`. The answer:

| field | example | meaning |
|---|---|---|
| `assign_to` | `"network"` | **what to do**: a team key, or `"triage"` = leave it for a human |
| `group` | `"network"` | triago's best guess, even when it sends the ticket to triage |
| `auto` | `true` | sure enough to assign without a human |
| `priority` | `2` | 1 (critical) … 5 (lowest), from the ITIL impact × urgency table |
| `impact`, `urgency` | `"high"`, `"medium"` | `null` when the security gate tripped |
| `security_incident` | `false` | `true` → page your SOC |
| `security_p`, `security_trip` | `0.006`, `0.05` | calibrated incident probability and the threshold it was compared with |
| `probabilities` | `{"group": {"network": 0.98, ...}, ...}` | every answer's probability, per question |
| `reason` | `"network=0.98, ..."` | one-line summary |
| `note` | `["[TRIAGO] group=network ...", ...]` | the explanation lines, ready to post as an internal note |
| `examples` | `[{"text": ..., "group": ...}]` | the similar past tickets the model was shown |
| `latency_ms` | `540` | time spent scoring |

Your rule then sets the assignment group from `assign_to` (map triago's keys to your group names), sets the
priority, and optionally posts `note` as an internal comment.

Team keys: `security`, `network`, `server_infra`, `end_user`, `identity_access`, `application`, `telephony`,
`facilities`, `other` (not IT). Change them in `triago/pipeline.py`.

Other endpoints: `GET /` (the web UI), `GET /info` (model, backend, adapter, thresholds, team names),
`GET /health`.

**Security:** the server refuses to listen beyond localhost without `TRIAGO_API_KEY`. It handles one request
at a time (one model, ~1 s each). If it is reachable from other machines, put it behind HTTPS (a reverse proxy).

## Python connector

When your system cannot call out, let triago poll it. Write a class with two methods
(the contract is in `triago/connector.py`):

```python
# mycompany/zammad.py: a sketch of the shape, check your system's API docs
import os, requests
from triago.connector import Ticket

GROUPS = {"network": "Network", "end_user": "Workplace", "security": "SOC"}  # triago key -> your group name

class Zammad:
    def __init__(self):
        self.url = os.environ["ZAMMAD_URL"]
        self.h = {"Authorization": f"Token token={os.environ['ZAMMAD_TOKEN']}"}

    def pending(self):                  # tickets that still need a team
        rows = requests.get(f"{self.url}/api/v1/tickets/search?query=state.name:new", headers=self.h).json()
        return [Ticket(t["id"], t["title"], t.get("body", "")) for t in rows]

    def apply(self, ticket, d, note):   # d.auto False -> your triage queue
        group = GROUPS.get(d.group, "Service Desk") if d.auto else "Service Desk"
        requests.put(f"{self.url}/api/v1/tickets/{ticket.id}", headers=self.h,
                     json={"group": group, "article": {"body": note, "internal": True}})
```

```bash
python -m triago.dispatcher run --connector mycompany.zammad:Zammad
```

`triago/glpi.py` is a complete, tested example of the same two methods for GLPI.

## Configuration

All settings are environment variables; all are optional.

| variable | default | what it does |
|---|---|---|
| `TRIAGO_BACKEND` | `mlx` on Apple Silicon, else `torch` | where the model runs |
| `TRIAGO_TORCH_DEVICE` | best available | `cuda`, `mps` or `cpu` (PyTorch backend) |
| `TRIAGO_CPU_BF16` | off | `1` = bfloat16 on CPU (less memory, needs a recent CPU) |
| `TRIAGO_MODEL` | Qwen3-4B-Instruct-2507 (4-bit for MLX) | any chat model whose answer letters are single tokens (checked at start) |
| `TRIAGO_ADAPTER` | none | a fine-tuned adapter folder from `python -m triago.finetune` |
| `TRIAGO_HOST`, `TRIAGO_PORT` | `127.0.0.1`, `8642` | where the server listens |
| `TRIAGO_API_KEY` | none | required as `Authorization: Bearer ...` when set |
| `GLPI_URL` | `http://localhost:8080/apirest.php` | GLPI REST API (dispatcher, feedback export) |
| `GLPI_USER`, `GLPI_PASSWORD` | `glpi`, `glpi` | GLPI account used by the dispatcher |
| `TRIAGO_URL` | `http://host.docker.internal:8642` | where the **GLPI plugin** finds triago (set in GLPI's environment) |
| `TRIAGO_TIMEOUT` | `4` | seconds the GLPI plugin waits before giving up |

Changing the model or backend changes how confident it is: re-run `python -m triago.bench` to re-fit the
calibration (see the [customization guide](customize.md)).
