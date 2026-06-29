#!/usr/bin/env bash
# Install NanoSAM inside the running nanoowl container and build its TRT engines.
# All artifacts go under /data (host: ~/jetson-containers/data) so they persist
# across the container's --rm restarts. Idempotent: safe to re-run.
#
# Run INSIDE the nanoowl container after `jetson-containers run $(autotag nanoowl)`.
set -e

NANOSAM_DIR=/data/nanosam
DATA_DIR=$NANOSAM_DIR/data
ENCODER_ONNX=$DATA_DIR/resnet18_image_encoder.onnx
DECODER_ONNX=$DATA_DIR/mobile_sam_mask_decoder.onnx
ENCODER_ENGINE=$DATA_DIR/resnet18_image_encoder.engine
DECODER_ENGINE=$DATA_DIR/mobile_sam_mask_decoder.engine
TRTEXEC=/usr/src/tensorrt/bin/trtexec

echo "== clone nanosam (skip if present) =="
[ -d "$NANOSAM_DIR" ] || git clone https://github.com/NVIDIA-AI-IOT/nanosam "$NANOSAM_DIR"

echo "== make nanosam importable =="
# torch2trt + transformers are already in dustynv/nanoowl; nanosam vendors what it needs.
# `setup.py develop --user` writes to ~/.local which is wiped on container --rm, so we
# also export PYTHONPATH=/data/nanosam in container_setup.sh as the durable path.
if ! python3 -c "import nanosam" 2>/dev/null; then
    (cd "$NANOSAM_DIR" && python3 setup.py develop --user)
fi

echo "== download model weights (skip if present) =="
mkdir -p "$DATA_DIR"
# Mask decoder ONNX — NVIDIA Box mirror (same URL jetson-containers uses).
[ -f "$DECODER_ONNX" ] || wget -O "$DECODER_ONNX" \
    https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx
# Image encoder ONNX — community mirror; the upstream Google Drive link is dead
# (nanosam issue #41, Nov 2025).
[ -f "$ENCODER_ONNX" ] || wget -O "$ENCODER_ONNX" \
    https://raw.githubusercontent.com/johnnynunez/nanosam/main/data/resnet18_image_encoder.onnx

echo "== build TensorRT engines (skip if present) =="
[ -f "$ENCODER_ENGINE" ] || "$TRTEXEC" \
    --onnx="$ENCODER_ONNX" \
    --saveEngine="$ENCODER_ENGINE" \
    --fp16

# Decoder needs explicit shape ranges (it takes a variable number of prompt points).
# Heads-up: on TRT 10.x the bundled decoder ONNX can fail to parse (nanosam issue #40,
# `IIOneHotLayer cannot be used to compute a shape tensor`). If trtexec errors below,
# re-export with: python3 -m nanosam.tools.export_sam_mask_decoder_onnx \
#   --model-type vit_t --checkpoint data/mobile_sam.pt --output $DECODER_ONNX \
#   --opset 17 --return-single-mask --gelu-approximate
[ -f "$DECODER_ENGINE" ] || "$TRTEXEC" \
    --onnx="$DECODER_ONNX" \
    --saveEngine="$DECODER_ENGINE" \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10

echo "== done =="
ls -lh "$ENCODER_ENGINE" "$DECODER_ENGINE"
