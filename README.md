# jetson-sam-demo

Live, text-prompted segmentation demo (NanoOWL → NanoSAM) on NVIDIA Jetson, for trade-show /
factory-AOI demos. Clone to any Jetson (JetPack 6.x / L4T R36.x) and run.

UI is the **ConanAI SAM WebUI** — a fork of NVIDIA's
[live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) (Apache 2.0). WebRTC + aiohttp
plumbing, NVIDIA dark theme, and GPU monitor are reused; the VLM inference module is replaced with
an in-process NanoOWL + NanoSAM service. See `webui/NOTICE.txt`.

## What it does
Type an object name in a browser → the Jetson detects (NanoOWL/OWL-ViT) and segments (NanoSAM) it
live, in real time, at the edge. No training required. Pitch: *"stand up a new inspection in 30
seconds — no dataset."*

## Quick start on a fresh Jetson

```bash
git clone <your-repo-url> jetson-sam-demo
cd jetson-sam-demo

# 1. Host bootstrap: Docker, NVIDIA runtime, the libnvdla fix, jetson-containers.
bash setup_host.sh
# log out / back in (docker group), then verify:
docker info | grep -i runtime         # expect 'nvidia'
docker run --rm hello-world

# 2. Stage scripts and the webui where the container can see them (host /data mount)
mkdir -p ~/jetson-containers/data/scripts
cp scripts/* ~/jetson-containers/data/scripts/
rm -rf ~/jetson-containers/data/webui
cp -r  webui    ~/jetson-containers/data/

# 3. Enter the NanoOWL container and run everything
jetson-containers run $(autotag nanoowl)
#   inside the container:
bash /data/scripts/container_setup.sh
```

Then open **https://<jetson-ip>:7860** from any PC on the LAN (accept the self-signed cert) and
type a prompt, e.g. `[an owl, a glove, a frog]`. Boxes + masks render on the live video.

## Files
- `setup_host.sh` — idempotent host setup. Encodes the **libnvdla_compiler.so** fix (auto-matched to
  your `nvidia-l4t-core` version) that minimized flashes need.
- `webui/` — the **ConanAI SAM WebUI** (forked from live-vlm-webui). aiohttp + WebRTC + static HTML
  serving the live OWL+SAM-annotated camera stream on `:7860`.
  - `owl_sam_service.py` — in-process inference (NanoOWL detect → NanoSAM segment).
  - `server.py` / `video_processor.py` — adapted from live-vlm-webui to swap in our service.
  - `static/index.html` — frontend (NVIDIA dark theme retained, prompt textarea repurposed for
    OWL tree expressions like `[an owl, a glove]`).
  - `NOTICE.txt` — Apache 2.0 attribution to upstream.
- `scripts/install_nanosam.sh` — clones NanoSAM, downloads weights, builds the SAM image-encoder +
  mask-decoder TRT engines into `/data/nanosam/data/`.
- `scripts/install_webui.sh` — pip-installs aiortc / av / aiohttp-cors / jtop into the container.
- `scripts/container_setup.sh` — builds engines, installs NanoSAM, installs webui deps, launches
  the webui on `:7860`.
- `scripts/make_test_video.py` — builds `/data/test.mp4` from the container's sample images (legacy;
  used by the older `tree_demo` fallback path).
- `scripts/patch_tree_demo.py`, `scripts/patch_tree_demo_sam.py` — legacy tree_demo patches; kept as
  a minimal-UI fallback if the webui breaks.
- `CLAUDE.md` — full project notes / state / next steps (also auto-loaded if you run Claude Code here).

## Notes
- Only `/data` (host: `~/jetson-containers/data`) persists across container runs — keep work there.
- Large artifacts (`*.engine`, `*.mp4`, `*.onnx`) are git-ignored; they're rebuilt per device.
- At the show, swap `--video /data/test.mp4` for a real camera: `--camera 0`.

See `CLAUDE.md` for the detailed build log, gotchas, and the roadmap (NanoSAM masks → AOI logic).
