#!/usr/bin/env bash
# Install the webui's Python deps into the running nanoowl container.
# The dustynv/nanoowl image already ships torch, torch2trt, opencv, cv2, transformers,
# nanoowl. We add aiortc + av + aiohttp-cors (WebRTC stack) and the small monitoring deps.
# Idempotent. Run INSIDE the nanoowl container.
set -e

echo "== install webui deps (pip) =="
# Use the canonical index — the image's custom indexes may not have current aiortc.
python3 -m pip install --index-url https://pypi.org/simple \
    "aiortc>=1.9.0" \
    "av>=12.3.0" \
    "aiohttp-cors>=0.7.0" \
    "nvidia-ml-py>=11.5.0" \
    "psutil>=5.9.8"

# jetson-stats (jtop) gives gpu_monitor.py access to Jetson tegrastats. Best-effort —
# it needs the host's jtop.service for full data; without it the GPU panel will fall
# back to nvidia-ml-py and still work.
python3 -c "import jtop" 2>/dev/null || \
    python3 -m pip install --index-url https://pypi.org/simple jetson-stats || \
    echo "(jetson-stats install failed — GPU monitor will be limited)"

echo "== verify imports =="
python3 -c "import aiortc, av, aiohttp_cors; print('webui deps OK')"
