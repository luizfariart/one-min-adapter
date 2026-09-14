#!/bin/bash
# Install the 1min.ai adapter: generate the launchd plist for this user and
# register the daemon so it survives reboots.
set -e

ADAPTER_DIR="$(cd "$(dirname "$0")" && pwd)"
HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python3"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LABEL="com.hermes.one-min-adapter"

# The plist ships with /Users/<user> placeholders; substitute this user's home.
sed -e "s#/Users/<user>#$HOME#g" \
    "$ADAPTER_DIR/com.hermes.one-min-adapter.plist" \
    > "$LAUNCH_AGENTS/$LABEL.plist"

echo "==> plist generated: $LAUNCH_AGENTS/$LABEL.plist"

# Confirm the API key file exists (created separately, chmod 600).
KEY_FILE="$HOME/.hermes/secrets/1min.key"
if [ ! -f "$KEY_FILE" ]; then
    echo "!! warning: API key not found at $KEY_FILE"
    echo "   create it with:  printf '%s' '<key>' > $KEY_FILE && chmod 600 $KEY_FILE"
else
    echo "==> API key present at $KEY_FILE"
fi

echo "==> Registering with launchd"
UID_NUM=$(id -u)
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$LAUNCH_AGENTS/$LABEL.plist"

echo "==> done. Verify:"
echo "  curl -s http://127.0.0.1:8400/health"
