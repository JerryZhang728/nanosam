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
rm -rf "$JC/data/webui"
cp -r "$REPO/webui" "$JC/data/"

cd "$JC"
TAG="$(./autotag nanoowl)"           # diagnostics go to stderr; only the tag on stdout
echo "== sam-demo: launching $TAG → container_setup.sh (web UI on :7860) =="
exec ./jetson-containers run "$TAG" bash /data/scripts/container_setup.sh "$@"
