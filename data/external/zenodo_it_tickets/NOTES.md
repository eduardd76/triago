# Classification of IT Support Tickets (Benitez Pereira, Zenodo 7648117)

- **Source:** https://zenodo.org/records/7648117 (DOI 10.5281/zenodo.7648117). HF mirror: https://huggingface.co/datasets/tasksource/it-support-tickets
- **Author:** Leonardo Santiago Benitez Pereira, Federal Institute of Santa Catarina (2022)
- **License:** CC BY 4.0 (attribution required in any published benchmark)
- **Real or synthetic:** REAL. Tickets come from an IT support company (MSP) in Florianópolis, Brazil. Three IT support professionals labeled them by hand in 2020. PII was removed with AWS Comprehend, regex and manual review, leaving placeholders such as `[NAME]`, `[TICKET ID]`, `[LOCATION]`, `[SERVER]`.
- **Downloaded:** from Zenodo on 2026-10-13, data files only: `X_train.csv`, `X_test.csv` (`id,text`), `y_train.csv`, `y_test.csv` (`id,category_truth`), `distributions.txt`, `metrics.txt`. I deliberately did NOT download `code.py`, `*.pickle` or `*.h5`. Pickle files can run code when loaded.
- **Rows:** 2,229 in total (1,572 train, 657 test). Every row joins by `id`.
- **Languages:** a mix of English, Portuguese, Spanish and German, often within the same ticket queue.
- **Text:** the subject is concatenated with the body and there is no separator. The median length is 72 chars and the max is 1,856. Many texts are subject-only stubs, e.g. `[TICKET ID] - New Support Ticket received - <subject>`.

## Label distribution (`category_truth`, train+test)

| label | count |
|---|---:|
| Support general | 762 |
| Fileservice | 684 |
| Software | 222 |
| O365 | 219 |
| Active Directory | 133 |
| Computer-Services | 106 |
| EOL | 103 |

## Caveats

- **745 of 2,229 texts are exact duplicates.** This is mostly templated tickets: 578 of the 684 Fileservice rows are "File Share Access" service requests, and all 103 EOL rows are the same automated Nexthink end-of-life notice. Deduplicate before benchmarking, or report results with and without duplicates.
- The authors' own model in `metrics.txt` reached 0.79 accuracy and 0.72 macro-F1 on the test split. Use that as a reference baseline.
- "Support general" is a catch-all queue. It contains network, mobile, OS-update, account-creation and mailbox-access tickets, so it does not map to one of our groups.
- Labels are an MSP's internal queues, not our taxonomy. The mapping below is approximate.

## Five representative texts (truncated)

1. `Fileservice`: "File Share Access - [TICKET ID] -[LOCATION]ServiceService Request: File Share Access  Grant access to file share for the following user: [NAME] (Username: [USERNAME]) Access Type: (RO) Read only…"
2. `O365`: "duplicate emails sentWe have two separate reports of duplicate messages being received by users. Both [NAME] and [NAME] appear to have sent duplicate messages…"
3. `Active Directory`: "[TICKET ID] - New Support Ticket received - [NAME] MS Account Login failure"
4. `Computer-Services`: "[TICKET ID] - New Support Ticket received - instalacion de impresora parral"
5. `Support general`: "[TICKET ID] - A Support Ticket was forwarded to your role - check_mk SD Hostalarm: ([LOCATION]-WSUS01) [COMPANY] [SERVER] -[LOCATION]Server - UP -> DOWN"

## Proposed mapping to JEV groups

| their label | JEV group | confidence / note |
|---|---|---|
| Active Directory | identity_access | clean (logins, AD groups, accounts) |
| Fileservice | identity_access (most rows) | ~85% are "File Share Access" grant requests, which go to identity_access under our tie-breaker. The remainder are share or capacity problems, which belong in server_infra. Per-row relabel needed for a strict benchmark. |
| O365 | application | mostly clean (Outlook, Exchange, Project Online). A few telephony or Teams-phone rows may exist. |
| Software | application / end_user | MIXED. A request to install an app on one PC belongs in end_user, while app errors and licenses belong in application. Treat as UNMAPPED, or relabel each row. |
| Computer-Services | end_user | mostly clean (PC reinstall, printers, scanners) |
| EOL | UNMAPPED | an automated asset-lifecycle (Nexthink) notice. The closest fit is end_user, but it is a templated bot ticket and should probably be excluded. |
| Support general | UNMAPPED | catch-all (network, mobile, OS updates, mailbox access, user creation). Do not force it into a group. |

**Suggested use:** a real-world sanity check on the 4–5 cleanly mappable labels, after deduplication. Alternatively, hand-relabel a deduplicated sample (about 1,480 unique texts) into our 9 groups.
