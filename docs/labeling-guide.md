# Ticket labeling guide (triago demo taxonomy)

Every ticket gets exactly one group (`g`), plus impact (`i`) and urgency (`u`).

## Groups

| key | team | owns |
|---|---|---|
| `security` | Security Operations (SOC) | Actual or likely incidents: phishing link clicked or credentials/QR code entered, malware/ransomware, suspicious sign-in or MFA change the user did not make, lost or stolen device or bag with a company device, data sent to the wrong recipient or shared publicly, leaked passwords, EDR/SIEM alerts. A user asking whether a *suspicious artefact* (email, link, call, Teams message, unexpected sign-in or MFA alert) is malicious is still `security`, even without clicking. Routine events are NOT security: a forgotten/expired password, a locked account, MFA re-setup on a new phone, a BitLocker recovery prompt after an update, a user wondering if a normal password-expiry prompt means they were hacked. |
| `network` | Network | Site-wide or multi-user wifi/LAN/WAN, VPN connectivity, firewall rules and blocked sites/ports, proxy, DNS (incl. new DNS records), DHCP, IP/VLAN requests, load balancers, one branch can't reach a system that others can. |
| `server_infra` | Server & Infrastructure | Servers, VMs, storage/SAN/NAS, file servers and shares (create, restore, capacity, unreachable share), backup, databases, cloud (Azure/AWS), certificates on websites/portals, monitoring alerts about servers, Kubernetes, datacenter hardware. |
| `end_user` | End-user Computing | One person's laptop/desktop and its hardware (disk, battery, wifi adapter, Windows Hello), OS updates, BitLocker, macOS, printers, monitors, docks, peripherals, installing software on a PC, Office apps crashing locally, new device setup. |
| `identity_access` | Identity & Access | Passwords, lockouts, MFA problems (not suspicious), AD/Entra accounts and groups, any request to grant/remove access or permissions (folders, SharePoint, SSH, sudo, admin rights), guest and service accounts incl. expiry, application test/training accounts, joiner/mover/leaver. |
| `application` | Business Applications | ERP/SAP functionality, CRM, Exchange/Outlook mailboxes and mail flow, distribution lists, Teams, SharePoint/OneDrive sync and sites, internal web apps, software licenses, application errors and feature requests. |
| `telephony` | Telephony & Mobile | Desk phones, softphones, Teams Phone/PSTN calling, contact center queues/agents/greetings, voicemail, company mobile phones and SIMs, mobile email setup, MDM/Company Portal. |
| `facilities` | Facilities | Meeting-room AV, badges and doors, power, lights, HVAC, desks, parking, EV chargers, physical moves. |
| `other` | not IT / no action | HR, payroll, vacation, expenses, travel booking, benefits, canteen, non-IT purchasing and printing services, empty or meaningless tickets, auto-replies, meeting invitations, bounce messages. |

Tie-breakers: route to the team that will *do the work*. Access/permissions → `identity_access` even if the target is a server or app. A device problem of one person → `end_user`; the same symptom for a whole floor/site → `network` or `server_infra`.

Settled boundary rules (decided 2026-10-13, apply exactly):
- Office app (Outlook, Excel, Word) crashing or misbehaving on ONE PC → `end_user`; mailbox, mail flow or server-side Exchange problems → `application`.
- OneDrive/SharePoint sync problems → `application`, even on one device.
- Physical access (badge zones, doors, turnstiles) → `facilities`; logical access (accounts, folders, systems) → `identity_access`.
- VPN or app login failing with an authentication error right after a password change → `identity_access`; VPN that connects but drops or is slow → `network`.
- Service-account password or app-secret expiry (including automated expiry notices) → `identity_access`.
- Environmental alerts in server rooms (temperature, humidity, CRAC/HVAC, UPS power) → `facilities`; the servers' own alerts → `server_infra`.
- Shared rugged devices (warehouse scanners) with an account error → `identity_access`; with a hardware/device error → `end_user`.

## Impact (`i`) and urgency (`u`)

- impact: `high` = many users, a whole site/department, or a business-critical service; `medium` = a team or small group; `low` = one person.
- urgency: `high` = work stopped now, no workaround (or a deadline within hours); `medium` = degraded, workaround exists; `low` = request or question that can wait.

## File format

One JSON object per line:
`{"t": "<subject> | <body>", "g": "<group key>", "i": "high|medium|low", "u": "high|medium|low", "hard": true|false, "topic": "<2-4 words>", "src": "<generator name>"}`

`hard` = a human dispatcher could plausibly hesitate between two groups.
