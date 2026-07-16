#!/usr/bin/env bash
# sam-demo — stage this repo into /data and launch the NanoOWL→NanoSAM live demo
# (ConanAI SAM WebUI) in ONE command. Stages scripts/ + webui/ into the container's
# /data mount, enters the nanoowl container, and runs container_setup.sh — which
# builds engines (if missing) and serves the web UI on :7860 (HTTPS, self-signed).
#
#   sam-demo                 # start the demo, then open https://<jetson-ip>:7860
#   sam-demo <extra args>    # forwarded to container_setup.sh
#
# jetson-containers checkout location (override if you moved it):
#   JETSON_CONTAINERS_DIR=/some/path sam-demo
set -euo pipefail

# Resolve the repo root even when invoked via the /usr/local/bin/sam-demo symlink.
SELF="$(realpath "$0")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
JC="${JETSON_CONTAINERS_DIR:-$HOME/Public/jetson-containers}"

if [ ! -x "$JC/jetson-containers" ] || [ ! -x "$JC/autotag" ]; then
    echo "sam-demo: jetson-containers not found at '$JC'." >&2
    echo "          Run setup_host.sh first, or set JETSON_CONTAINERS_DIR to your checkout." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "sam-demo: Docker isn't usable in this shell." >&2
    echo "          If you JUST ran setup_host.sh, activate the docker group once:" >&2
    echo "              newgrp docker      # (or log out / back in)" >&2
    echo "          then run 'sam-demo' again." >&2
    exit 1
fi

# Stage the current repo into the container's /data mount (always fresh, so edits to
# webui/ or scripts/ are picked up without a manual copy).
echo "== staging scripts + webui from $REPO into $JC/data =="
mkdir -p "$JC/data/scripts"
cp "$REPO"/scripts/* "$JC/data/scripts/"
# The container runs as root and may have left root-owned __pycache__/*.pyc under the
# staged webui/, which this (non-root) command can't delete. Fall back to sudo for the
# wipe; container_setup.sh now sets PYTHONDONTWRITEBYTECODE=1 so new runs stop creating
# them, meaning the sudo path is only ever hit once (to clear pre-existing leftovers).
rm -rf "$JC/data/webui" 2>/dev/null || sudo rm -rf "$JC/data/webui"
cp -r "$REPO/webui" "$JC/data/"

# Ensure the dir the UI file-browser defaults to (/data/videos) always exists, and seed it
# with the repo's sample clip(s). -n so we never clobber videos the user dropped in there.
mkdir -p "$JC/data/videos"
[ -d "$REPO/videos" ] && cp -rn "$REPO"/videos/* "$JC/data/videos/" 2>/dev/null || true

# Ensure the host jtop.service is up so /run/jtop.sock EXISTS before the container starts.
# jetson-containers only adds "-v /run/jtop.sock:/run/jtop.sock" when the socket already
# exists, so without this the web UI's GPU/VRAM/CPU panel stays empty (typically after a
# reboot, when the container is launched before the service comes up). Best-effort, non-fatal.
if [ ! -S /run/jtop.sock ]; then
    if systemctl cat jtop.service >/dev/null 2>&1; then
        echo "== ensuring host jtop.service is up (GPU/VRAM/CPU panel needs /run/jtop.sock) =="
        sudo systemctl start jtop.service 2>/dev/null || true
        for _ in $(seq 1 10); do
            if [ -S /run/jtop.sock ]; then break; fi
            sleep 1
        done
    fi
    if [ ! -S /run/jtop.sock ]; then
        echo "sam-demo: WARN /run/jtop.sock missing -> GPU/VRAM/CPU panel may stay empty." >&2
        echo "          Fix: run setup_host.sh (installs jetson-stats + enables jtop.service)." >&2
    fi
fi

# Pin the container's jetson-stats to the HOST's version. The jtop client refuses to talk
# to a different-version jtop service, which silently empties the GPU/VRAM/CPU panel.
# install_webui.sh reads this file and installs the matching version.
HOST_JTOP_VER="$(jtop --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
if [ -z "${HOST_JTOP_VER:-}" ]; then
    HOST_JTOP_VER="$(pip3 show jetson-stats 2>/dev/null | awk '/^Version:/{print $2}' || true)"
fi
if [ -n "${HOST_JTOP_VER:-}" ]; then
    echo "== host jetson-stats $HOST_JTOP_VER -> container will pin jtop to match =="
    echo "$HOST_JTOP_VER" > "$JC/data/.jtop_version"
else
    rm -f "$JC/data/.jtop_version" 2>/dev/null || true
fi

cd "$JC"
TAG="$(./autotag nanoowl)"           # diagnostics go to stderr; only the tag on stdout
echo "== sam-demo: launching $TAG → container_setup.sh (web UI on :7860) =="
exec ./jetson-containers run "$TAG" bash /data/scripts/container_setup.sh "$@"
