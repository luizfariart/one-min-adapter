#!/bin/bash
# Install both daemons: the model router (:8400) and the portal OAuth proxy
# (:8645). Generates launchd plists for this user and registers them.
set -e

ADAPTER_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
UID_NUM=$(id -u)

# --- model router (:8400) ---
ROUTER_LABEL="com.hermes.one-min-adapter"
sed -e "s#/Users/<user>#$HOME#g" \
    "$ADAPTER_DIR/com.hermes.one-min-adapter.plist" \
    > "$LAUNCH_AGENTS/$ROUTER_LABEL.plist"
echo "==> router plist: $LAUNCH_AGENTS/$ROUTER_LABEL.plist"

# --- portal OAuth proxy (:8645) ---
PORTAL_LABEL="com.hermes.portal-proxy"
sed -e "s#/Users/<user>#$HOME#g" \
    "$ADAPTER_DIR/com.hermes.portal-proxy.plist" \
    > "$LAUNCH_AGENTS/$PORTAL_LABEL.plist"
echo "==> portal proxy plist: $LAUNCH_AGENTS/$PORTAL_LABEL.plist"

# API key check (router needs it for the 1min.ai backend)
KEY_FILE="$HOME/.hermes/secrets/1min.key"
if [ ! -f "$KEY_FILE" ]; then
    echo "!! warning: API key not found at $KEY_FILE"
    echo "   create it with:  printf '%s' '<key>' > $KEY_FILE && chmod 600 $KEY_FILE"
else
    echo "==> API key present at $KEY_FILE"
fi

# Register both (bootout first to apply env-var changes cleanly)
for LABEL in "$ROUTER_LABEL" "$PORTAL_LABEL"; do
    launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
done
sleep 1
for LABEL in "$ROUTER_LABEL" "$PORTAL_LABEL"; do
    launchctl bootstrap "gui/$UID_NUM" "$LAUNCH_AGENTS/$LABEL.plist"
    echo "==> registered: $LABEL"
done

echo
echo "Done. Verify:"
echo "  curl -s http://127.0.0.1:8400/health   # router"
echo "  curl -s http://127.0.0.1:8645/v1/models -H 'Authorization: Bearer x'  # portal proxy"
