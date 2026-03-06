#!/usr/bin/env bash
# install_quest.sh
# Installs the Quest VR teleop dependencies on top of the existing .venv_teleop.
# Run AFTER install_pico.sh (which creates .venv_teleop).
#
# Usage:  bash install_scripts/install_quest.sh   (run from repo root)
#
# What this does:
#   1. Activates .venv_teleop
#   2. Installs gear_sonic[quest]  (vuer + scipy)
#   3. Optionally generates a self-signed TLS cert for same-WiFi use
#      (needed when Quest and workstation are on the same network and
#       you want to avoid the ngrok tunnel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/.venv_teleop"

# ── 1. Check .venv_teleop exists ─────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[ERROR] .venv_teleop not found at $VENV"
    echo "        Run install_scripts/install_pico.sh first."
    exit 1
fi

echo "[INFO] Activating .venv_teleop …"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Install quest extra ────────────────────────────────────────────────────
echo "[INFO] Installing gear_sonic[quest] (vuer, scipy) …"
uv pip install -e "$REPO_ROOT/gear_sonic[quest]"

# ── 3. Optional: self-signed TLS cert ────────────────────────────────────────
# vuer requires HTTPS for WebXR. When using the ngrok tunnel (default) this is
# handled automatically. If you prefer a direct same-WiFi connection without
# ngrok, generate a cert here and pass ngrok=False to OpenTeleVision.
CERT="$REPO_ROOT/cert.pem"
KEY="$REPO_ROOT/key.pem"

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
    echo "[INFO] Generating self-signed TLS certificate for same-WiFi use …"
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$KEY" -out "$CERT" \
        -days 365 -nodes \
        -subj "/CN=sonic-quest-bridge" \
        2>/dev/null
    echo "[OK] cert.pem and key.pem written to repo root."
    echo "     Import cert.pem into the Quest browser's trusted roots, or"
    echo "     accept the security warning on first connection."
else
    echo "[SKIP] cert.pem / key.pem already exist — skipping generation."
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Quest teleop dependencies installed!"
echo ""
echo "  Run the bridge:"
echo "    source .venv_teleop/bin/activate"
echo "    python gear_sonic/scripts/quest_zmq_publisher.py"
echo ""
echo "  Then open the printed URL in the Meta Quest Browser."
echo "══════════════════════════════════════════════════════════════"
