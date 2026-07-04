#!/usr/bin/env bash
# sam-demo — launch the NanoOWL→NanoSAM live segmentation demo (ConanAI SAM WebUI)
# from anywhere, in ONE command. Enters the nanoowl container and runs
# container_setup.sh, which builds engines (if missing) and serves the web UI
# on :7860 (HTTPS, self-signed).
#
#   sam-demo                 # start the demo, then open https://<jetson-ip>:7860
#   sam-demo <extra args>    # forwarded to container_setup.sh
#
# jetson-containers checkout location (override with the env var if you move it):
#   JETSON_CONTAINERS_DIR=/some/path sam-demo
set -euo pipefail

JC="${JETSON_CONTAINERS_DIR:-$HOME/Public/jetson-containers}"

if [ ! -x "$JC/jetson-containers" ] || [ ! -x "$JC/autotag" ]; then
    echo "sam-demo: jetson-containers not found at '$JC'." >&2
    echo "          Set JETSON_CONTAINERS_DIR to your checkout, e.g.:" >&2
    echo "          JETSON_CONTAINERS_DIR=/path/to/jetson-containers sam-demo" >&2
    exit 1
fi

cd "$JC"
TAG="$(./autotag nanoowl)"           # diagnostics go to stderr; only the tag on stdout
echo "== sam-demo: launching $TAG → container_setup.sh (web UI on :7860) =="
exec ./jetson-containers run "$TAG" bash /data/scripts/container_setup.sh "$@"
