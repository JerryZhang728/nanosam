#!/usr/bin/env bash
# Build a stronger NanoOWL engine: google/owlvit-base-patch16.
# Same model size as patch32 but 4× finer spatial grid → much better small-object
# recall, similar speed (~5-8 fps on Orin NX). Run INSIDE the nanoowl container.
# One-time, ~10 min. After this, container_setup.sh auto-prefers this engine.
#
# Heavier option: google/owlvit-large-patch14 — best accuracy, ~1-2 fps. Set
# MODEL=google/owlvit-large-patch14 ENGINE=/data/owl_image_encoder_large.engine
# in env before running this script.
set -e

MODEL=${MODEL:-google/owlvit-base-patch16}
ENGINE=${ENGINE:-/data/owl_image_encoder_b16.engine}

if [ -f "$ENGINE" ]; then
    echo "$ENGINE already exists. Delete it to rebuild."
    ls -lh "$ENGINE"
    exit 0
fi

echo "== building NanoOWL engine =="
echo "   model:  $MODEL"
echo "   engine: $ENGINE"
echo "   (~10 min on Orin NX; longer for large-patch14)"
python3 -m nanoowl.build_image_encoder_engine "$ENGINE" --model_name "$MODEL"
ls -lh "$ENGINE"
echo "== done. Re-run container_setup.sh — it'll auto-pick this engine. =="
