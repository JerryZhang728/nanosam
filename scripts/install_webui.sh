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
# The jtop CLIENT must match the host's jtop SERVICE version exactly, or it refuses to
# connect ("Mismatch version jtop service: [x] and client: [y]") and the GPU/VRAM panel
# silently shows no data. run_demo.sh writes the host's version to /data/.jtop_version.
# NOTE: do NOT skip when "import jtop" already succeeds - the base image often ships a
# NEWER jetson-stats than the host, which is exactly the mismatch we need to correct.
JTOP_VER="$(tr -d '[:space:]' < /data/.jtop_version 2>/dev/null || true)"
JTOP_CUR="$(python3 -c 'import jtop; print(jtop.__version__)' 2>/dev/null || true)"
if [ -n "$JTOP_VER" ]; then
    if [ "$JTOP_CUR" = "$JTOP_VER" ]; then
        echo ">> jetson-stats $JTOP_CUR already matches host jtop.service"
    else
        echo ">> pinning jetson-stats to host version $JTOP_VER (container had ${JTOP_CUR:-none})"
        python3 -m pip install --index-url https://pypi.org/simple "jetson-stats==$JTOP_VER" || \
            echo "(jetson-stats==$JTOP_VER install failed - GPU/VRAM panel may stay empty)"
    fi
else
    # No host version file - host jetson-stats probably not installed. Best effort, unpinned.
    python3 -c "import jtop" 2>/dev/null || \
        python3 -m pip install --index-url https://pypi.org/simple jetson-stats || \
        echo "(jetson-stats install failed - GPU monitor will be limited)"
fi

# supervision => ByteTrack, for the "NanoOWL + ByteTrack" mode. Install --no-deps: the image
# already has a CUDA-built cv2 and numpy 1.x; supervision's default deps pull a preview
# opencv/numpy2 that clobbers them (native "double free" at runtime). scipy + the small pure
# deps below are safe. If this fails, OwlSamService just falls back to plain boxes.
if ! python3 -c "import supervision" 2>/dev/null; then
    python3 -m pip install --index-url https://pypi.org/simple --no-deps "supervision==0.29.1" \
      && python3 -m pip install --index-url https://pypi.org/simple scipy defusedxml "pydeprecate<0.10" \
      || echo "(supervision install failed — NanoOWL+ByteTrack falls back to plain boxes)"
fi

echo "== verify imports =="
python3 -c "import aiortc, av, aiohttp_cors; print('webui deps OK')"
python3 -c "import supervision; print('supervision (ByteTrack) OK', supervision.__version__)" \
    || echo "(supervision not importable — ByteTrack mode will fall back to plain boxes)"
