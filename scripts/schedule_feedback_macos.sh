#!/bin/sh
# Nightly GLPI feedback export on macOS (launchd). Human reassignments become examples by morning;
# the running server and dispatcher reload them on their next ticket.
#
#   scripts/schedule_feedback_macos.sh            # install: every night at 02:00 (or on wake if asleep)
#   scripts/schedule_feedback_macos.sh 23 30      # install: every night at 23:30
#   scripts/schedule_feedback_macos.sh --remove   # uninstall
#
# Log: logs/feedback.log in the repo. Linux: use cron, e.g.
#   0 2 * * *  cd /path/to/triago && .venv/bin/python -m triago.dispatcher feedback >> logs/feedback.log 2>&1
set -eu

LABEL=io.github.triago.feedback
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [ "${1:-}" = "--remove" ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $LABEL"
    exit 0
fi

HOUR=${1:-2}
MINUTE=${2:-0}
[ -x "$REPO/.venv/bin/python" ] || { echo "no $REPO/.venv: create it first (see README)"; exit 1; }
mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string><string>-c</string>
        <string>echo "--- \$(date '+%F %T')"; exec "$REPO/.venv/bin/python" -m triago.dispatcher feedback</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>
    <key>StandardOutPath</key><string>$REPO/logs/feedback.log</string>
    <key>StandardErrorPath</key><string>$REPO/logs/feedback.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
printf 'scheduled %s daily at %02d:%02d -> %s/logs/feedback.log\n' "$LABEL" "$HOUR" "$MINUTE" "$REPO"
echo "run it now once:  launchctl kickstart gui/$(id -u)/$LABEL"
