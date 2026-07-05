#!/usr/bin/env bash
# Run INSIDE the nanoowl container (after `jetson-containers run $(autotag nanoowl)`).
# Builds the OWL engine, installs NanoSAM, then launches the new ConanAI SAM WebUI
# (fork of live-vlm-webui) on :7860 with WebRTC + HTTPS.
# Assumes this repo's scripts/ and webui/ are staged at /data/scripts and /data/webui
# (see README for how to stage them on the host).
set -e

ENGINE=/data/owl_image_encoder_patch32.engine
B16_ENGINE=/data/owl_image_encoder_b16.engine
LARGE_ENGINE=/data/owl_image_encoder_large.engine
SCRIPTS=/data/scripts
WEBUI=/data/webui

echo "== install base python deps =="
python3 -c "import aiohttp" 2>/dev/null || pip3 install aiohttp --index-url https://pypi.org/simple

echo "== build OWL-ViT base/patch32 TensorRT engine (default; skip if exists) =="
[ -f "$ENGINE" ] || python3 -m nanoowl.build_image_encoder_engine "$ENGINE"

# Auto-prefer stronger engines if the user has built them:
#   build_owl_b16.sh    → /data/owl_image_encoder_b16.engine    (better recall, ~5-8 fps)
#   build_owl_b16.sh with MODEL=google/owlvit-large-patch14 ENGINE=/data/owl_image_encoder_large.engine
# The OWL model name MUST match the engine's patch geometry or the decoder will
# crash with a shape mismatch (patch32→576 tokens, patch16→2304, large→3600).
OWL_MODEL="google/owlvit-base-patch32"
if [ -f "$LARGE_ENGINE" ]; then
    echo "== detected owlvit-large engine — using it (best accuracy) =="
    ENGINE="$LARGE_ENGINE"
    OWL_MODEL="google/owlvit-large-patch14"
elif [ -f "$B16_ENGINE" ]; then
    echo "== detected owlvit-base-patch16 engine — using it =="
    ENGINE="$B16_ENGINE"
    OWL_MODEL="google/owlvit-base-patch16"
fi

echo "== install NanoSAM + build its TRT engines (idempotent) =="
bash "$SCRIPTS/install_nanosam.sh"

echo "== install webui deps (aiortc, av, aiohttp-cors, jtop) =="
bash "$SCRIPTS/install_webui.sh"

echo "== verify webui has been staged =="
if [ ! -f "$WEBUI/server.py" ]; then
    echo "ERROR: $WEBUI/server.py missing. On the host, run:"
    echo "  cp -r webui/ ~/Public/jetson-containers/data/"
    exit 1
fi

# Old test video / tree_demo path is no longer used by the webui, but harmless if present.
[ -f /data/test.mp4 ] || python3 "$SCRIPTS/make_test_video.py" || true

# Make nanosam + the webui package importable. /data/nanosam holds nanosam (cloned
# there by install_nanosam.sh because ~/.local is wiped on container --rm); /data
# is the parent so `python3 -m webui.server` finds the webui/ package.
export PYTHONPATH=/data:/data/nanosam:${PYTHONPATH:-}

echo "== launch ConanAI SAM WebUI on :7860 =="
echo "   open https://<jetson-ip>:7860  (self-signed cert — accept the warning)"
echo "   prompt example: [an owl, a glove, a frog]"
cd /data
exec python3 -m webui.server \
    --port 7860 \
    --owl-engine "$ENGINE" \
    --owl-model "$OWL_MODEL" \
    --sam-encoder /data/nanosam/data/resnet18_image_encoder.engine \
    --sam-decoder /data/nanosam/data/mobile_sam_mask_decoder.engine \
    --prompt "[a person]" \
    --process-every 6
