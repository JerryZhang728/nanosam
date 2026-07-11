# nanosam

Live, text-prompted segmentation demo (NanoOWL → NanoSAM) on NVIDIA Jetson, for trade-show /
factory-AOI demos. Clone to any Jetson (JetPack 6.x / L4T R36.x) and run.

## TL;DR — fresh L4T Jetson
```bash
sudo apt install -y git && mkdir -p ~/Public && git clone https://github.com/JerryZhang728/nanosam.git ~/Public/nanosam && cd ~/Public/nanosam
bash setup_host.sh      # sets up Docker, jetson-containers, and the 'sam-demo' command
newgrp docker           # fresh box only, one time (activates the docker group)
sam-demo                # stage + launch (first run builds engines — a few minutes)
# → open https://<jetson-ip>:7860   (or https://localhost:7860 on the Jetson itself)
```

UI is the **ConanAI SAM WebUI** — a fork of NVIDIA's
[live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) (Apache 2.0). WebRTC + aiohttp
plumbing, NVIDIA dark theme, and GPU monitor are reused; the VLM inference module is replaced with
an in-process NanoOWL + NanoSAM service. See `webui/NOTICE.txt`.

## What it does
Type an object name in a browser → the Jetson detects (NanoOWL/OWL-ViT) and segments (NanoSAM) it
live, in real time, at the edge. No training required. Pitch: *"stand up a new inspection in 30
seconds — no dataset."*

## Requirements
- An **NVIDIA L4T (Jetson)** host — tested on **Orin NX 16GB, L4T R36.4.x** (JetPack 6.2).
- **CUDA and TensorRT are NOT required on the host** — they ship inside the `dustynv/nanoowl`
  container. The host only needs base **L4T** (GPU driver + DLA libs); `setup_host.sh` adds Docker,
  the NVIDIA container runtime, and the `libnvdla_compiler.so` fix from the L4T apt repo.
- `setup_host.sh` runs a preflight that **fails fast with a clear message** if L4T isn't present, and
  installs any missing host basics (`git`, `curl`, `python3-pip`).

## Quick start on a fresh Jetson

**Step 0 — get the code.** A bare L4T flash may not ship `git`, so install it first (or download the
repo as a ZIP from GitHub):
```bash
sudo apt-get update && sudo apt-get install -y git    # skip if git is already present
mkdir -p ~/Public && git clone https://github.com/JerryZhang728/nanosam.git ~/Public/nanosam
cd ~/Public/nanosam
```

**Step 1 — set up the machine (once).** Installs Docker, the NVIDIA container runtime, the
`libnvdla_compiler.so` fix, jetson-containers, and the `sam-demo` launcher. Idempotent + resumable —
it fails fast with a clear message if this isn't an L4T host.
```bash
bash setup_host.sh
newgrp docker        # fresh box only: activate the docker group once (or log out / back in)
```

**Step 2 — run the demo.** `sam-demo` stages the repo into the container's `/data` and launches
everything (builds the TRT engines on first run).
```bash
sam-demo             # first run pulls the ~GB image + builds engines (a few minutes)
```

**Step 3 — open it.** From any PC on the LAN: **https://&lt;jetson-ip&gt;:7860** (or
**https://localhost:7860** on the Jetson itself). Accept the self-signed cert, then type a prompt like
`[an owl, a glove, a frog]` — boxes + masks render on the live video.

<details><summary>What <code>sam-demo</code> / <code>setup_host.sh</code> do under the hood (manual equivalent)</summary>

```bash
# stage repo -> container /data mount
mkdir -p ~/Public/jetson-containers/data/scripts
cp scripts/* ~/Public/jetson-containers/data/scripts/
rm -rf ~/Public/jetson-containers/data/webui
cp -r  webui  ~/Public/jetson-containers/data/
# enter the container and build + launch
cd ~/Public/jetson-containers
./jetson-containers run $(./autotag nanoowl) bash /data/scripts/container_setup.sh
```
</details>

## Files
- `setup_host.sh` — idempotent host setup: L4T preflight, Docker, NVIDIA runtime, the
  **libnvdla_compiler.so** fix (auto-matched to your `nvidia-l4t-core` version), jetson-containers, and
  it installs the **`sam-demo`** command (`/usr/local/bin/sam-demo` → `container/run_demo.sh`).
- `container/` — jetson-containers integration for this project. `run_demo.sh` (invoked as `sam-demo`)
  stages the repo into `/data` and launches the container + web UI; `data/` is the bind mount that
  becomes `/data` inside the container. See `container/README.md`.
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
- Only `/data` (host: `~/Public/jetson-containers/data`) persists across container runs — keep work there.
- Large artifacts (`*.engine`, `*.mp4`, `*.onnx`) are git-ignored; they're rebuilt per device.
- At the show, swap `--video /data/test.mp4` for a real camera: `--camera 0`.

See `CLAUDE.md` for the detailed build log, gotchas, and the roadmap (NanoSAM masks → AOI logic).
