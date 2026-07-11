# CLAUDE.md — nanosam project notes

> Living notes for the NanoOWL/NanoSAM live demo. Auto-loaded by Claude Code when run in this repo.
> This repo is meant to be **git-cloned to any new NVIDIA Jetson** to reproduce the demo. Keep these
> notes and the scripts updated as the project evolves.

## Goal
Live **text-prompted segmentation** demo for **Automation Taipei 2026** (Aug 19–22, Nangang Taipei).
Pitch: *"zero-training AOI — set up a new inspection in 30 seconds, no dataset."*
Pipeline: **NanoOWL** (text → boxes) → **NanoSAM** (box → mask), real-time on the Jetson, shown in a
browser web UI (the SAM analog of live-vlm-webui).

## Reference hardware / OS (first device)
- Board EEP-150N carrier, **Jetson Orin NX 16GB** (GPU compute 8.7), IP **192.168.8.16**, user `ubuntu`
- **JetPack 6.2 / L4T R36.4.4**, **CUDA 12.6**, TensorRT 10.4, Ubuntu 22.04 (glibc 2.35), ~179 GB free
- No camera attached → demo runs from a video file; a USB webcam will be used at the show.

## Reproducible setup (what the scripts encode)
Run `bash setup_host.sh` on a fresh Jetson. It is idempotent and does:
1. Install Docker; add user to docker group.
2. Install **nvidia-container-toolkit**; `nvidia-ctk runtime configure --runtime=docker`; restart docker.
3. Enable the **L4T apt repo** (uncomment `common` + `t234` in `nvidia-l4t-apt-source.list`).
4. **Fix the missing `libnvdla_compiler.so`** — the key gotcha. Minimized flashes ship
   `libnvdla_runtime.so` but not the compiler lib, so the container's TensorRT throws
   `ImportError: libnvdla_compiler.so: cannot open shared object file`. The script installs
   `nvidia-l4t-dla-compiler` **pinned to the same version as `nvidia-l4t-core`** (auto-detected).
5. Clone + `install.sh` **jetson-containers**.

Then, inside `jetson-containers run $(autotag nanoowl)`, run `bash /data/scripts/container_setup.sh`
to build the OWL engine, generate the test video, patch the demo, and launch the web UI on **:7860**.

## Status log
- [x] Host setup + Docker + nvidia runtime working.
- [x] libnvdla fix applied (lib present, listed in `drivers.csv`).
- [x] `dustynv/nanoowl:r36.4.0` pulled.
- [x] OWL-ViT TensorRT engine built → `/data/owl_image_encoder_patch32.engine`.
- [x] `owl_predict` validated (TensorRT inference OK; ignore the cosmetic read-only cv2 draw error).
- [x] Patched `/data/tree_demo` (`--video` + loop); `/data/test.mp4` generated from assets.
      Note: had to install `aiohttp` (use `--index-url https://pypi.org/simple`); the read-loop
      patch must match only the `if not re:`/`return re, None` pair (file has blank lines between stmts).
- [x] **Web demo CONFIRMED** — draws live boxes at http://192.168.8.16:7860 for typed prompts.
      Test feed dwell is controlled by FRAMES_PER_IMG in make_test_video.py (demo ignores video FPS).
- [x] **NanoSAM** mask overlay — verified live on Jetson (engines built, tree_demo at :7860 draws
      colored masks under OWL boxes). The patched `tree_demo` is now the **legacy fallback**.
- [x] **ConanAI SAM WebUI** (`webui/`) — fork of live-vlm-webui swapping VLMService → OwlSamService.
      WebRTC + aiohttp + NVIDIA dark theme reused; in-process NanoOWL + NanoSAM inference returns
      annotated frames that VideoProcessorTrack swaps into the outgoing track. Runs on `:7860`
      (HTTPS, self-signed cert). **VERIFIED live on the 2nd Jetson (192.168.8.6).**
- [x] **STABLE BASELINE (2026-07-11) — 3 selectable modes + offline UI.** This is a known-good rollback
      point (git tag `stable-3modes`). What's in it:
      • Model dropdown drives 3 modes: `nanoowl` (boxes), `nanoowl+bytetrack` (stable IDs, NEW),
        `nanoowl+nanosam` (masks). ByteTrack via `supervision` (installed `--no-deps` in
        install_webui.sh so it can't clobber the CUDA cv2 — that caused a native `double free`).
      • **Offline UI fix:** lucide/marked/dompurify were CDN `<script>` tags; the Jetson can't reach
        unpkg/jsdelivr, so the page's load event hung → dropdown stuck on "Loading models...". Vendored
        all three under `webui/static/vendor/` + added a `/vendor` static route. UI now needs no internet.
      • Sample clip `videos/person on railtrack.mp4` bundled; `run_demo.sh` seeds `/data/videos`.
      • README clones straight to `~/Public/nanosam` (setup_host relocation is a no-op when already there).
      • Perf note: the visible "lag" is `process_every` (boxes refresh only every Nth frame), NOT
        frame-dropping (that's off, `max_frame_latency=0`). Sweet spot N ≈ source_fps × inference_time
        (OWL/ByteTrack ~170ms→N≈3; OWL+SAM ~365ms→N≈5-6). Still inference-rate-limited between frames.
- [ ] **NEXT: 4th mode `nanoowl+bytetrack+nanosam`** — OWL→ByteTrack→SAM (tracked + segmented), mask-only
      render (no box/ID), option (b) motion-shifted mask (translate last mask by tracker delta between
      inferences so it follows the object; shape stale, position tracks). Masks can't coast like boxes —
      SAM must re-run — so between inferences only the box coasts, not the mask.
- [ ] **AOI logic** (ROI per slot, presence/count, PASS/FAIL + MISSING/MISPLACED overlay).
- [ ] **AOI logic** (ROI per slot, presence/count, PASS/FAIL + MISSING/MISPLACED overlay).
- [ ] (Optional) SAM → VLM chain.
- [ ] Booth polish: lighting, USB camera (`--camera 0`), fullscreen UI, pitch script.

## Gotchas (don't relearn these)
- **VRAM leak → OOM after ~1 min (FIXED — root cause was WebRTC, NOT the models).** Symptom: unified
  memory climbs ~5→14.8GB over a minute of live video, caps at 16GB, board throttles / container dies.
  **Root cause:** the `MediaRelay.subscribe()` calls in `server.py` used the default `buffered=True`,
  which gives each subscriber an *unbounded* frame queue. The video source decodes at native FPS (~25-30)
  but the Jetson encodes+sends only ~10 FPS to the browser; the ~15 FPS difference (raw ~6MB frames)
  piled up in that queue forever. **Fix:** `relay.subscribe(track, buffered=False)` on all 3 call sites
  (RTSP / local-file / webcam) — serve latest frame, drop the backlog (correct for a live demo anyway).
  **How we proved it:** `[mem]` logging in `owl_sam_service` showed `torch_alloc` DEAD FLAT at 1.05GB
  across hundreds of inferences while total RAM ran to 14.8GB → the leak was entirely outside PyTorch.
  Lesson: a timed OOM under live video is almost always frame-buffer backpressure, not the model. TOPS
  is irrelevant (that shows as low FPS, never a timed stop).
- Secondary/defensive (kept, but were NOT the leak): `_run_inference` runs under `torch.inference_mode()`
  (good hygiene; neither NanoOWL nor NanoSAM wraps its own forwards). `SAM_EMPTY_CACHE_EVERY` /
  `SAM_MEM_LOG_EVERY` env knobs remain for diagnostics; empty_cache defaults OFF (torch wasn't leaking).
- Container runs `--rm` → **only `/data` persists** (host `~/Public/jetson-containers/data`). Keep work there.
  The webui runs from the STAGED copy at `/data/webui/`; after editing `webui/` in the repo, re-copy it
  (`cp webui/*.py ~/Public/jetson-containers/data/webui/`) or the container won't see the change.
- **GPU/VRAM/CPU panel empty in the web UI (jtop/tegrastats gotcha — FIXED).** The UI's
  monitor reads Jetson stats via `jtop`, which needs the **host's** `jtop.service`
  (jetson-stats) running — that service is what creates `/run/jtop.sock`. `jetson-containers
  run.sh` only mounts the socket (`-v /run/jtop.sock:/run/jtop.sock`) **if it already exists**,
  and the in-container jetson-stats (from `install_webui.sh`) is only the *client*. If
  jetson-stats isn't installed on the host, there's no socket → nothing mounts → bars stay
  empty. `nvidia-smi` does NOT report GPU-util/VRAM on Tegra, so jtop is the only source.
  **Fix:** `setup_host.sh` step **[6/6]** now installs jetson-stats system-wide and
  enables/starts `jtop.service` on the host (mirrors the vlm project's fix). Verify on host:
  `systemctl is-active jtop.service` and `ls -l /run/jtop.sock`.
- **Repo location is self-correcting.** `setup_host.sh` relocates the checkout to
  `~/Public/nanosam` if it was cloned elsewhere (e.g. `~/nanosam`), so nanosam and the shared
  `~/Public/jetson-containers` always end up side-by-side under `~/Public`.
- Keep all L4T components on the **same version** (here 36.4.4).
- `tree_demo.py` loads `./index.html` relatively → launch from inside `/data/tree_demo`.
- `cv2.VideoCapture` takes a file path too; our patch loops it by seeking to frame 0 on EOF.

## NanoSAM integration (implemented; awaiting on-device verification)
**Path chosen:** add nanosam *inside* the existing nanoowl container (no second container, no IPC).
Same process feeds OWL boxes straight into SAM.

**What the new scripts do:**
- `scripts/install_nanosam.sh` — clones `NVIDIA-AI-IOT/nanosam` to `/data/nanosam`, downloads the two
  ONNX weights (from **stable mirrors** — upstream Drive link for the image encoder is dead, see
  nanosam issue #41), and builds both TRT engines via `/usr/src/tensorrt/bin/trtexec`.
- `scripts/patch_tree_demo_sam.py` — anchored string-patch over `/data/tree_demo/tree_demo.py`:
  imports `nanosam.utils.predictor.Predictor`, adds CLI args
  (`--sam_image_encoder_engine`, `--sam_mask_decoder_engine`, `--no_sam`, `--mask_alpha`),
  instantiates SAM after the TreePredictor, and inserts an `_overlay_sam_masks` step right BEFORE
  the existing `draw_tree_output` call.
- `container_setup.sh` calls both, then `export PYTHONPATH=/data/nanosam` before launching.

**SAM call pattern (do not regress):** `sam.set_image(image_pil)` **once per frame** (heavy ResNet18
encoder), then `sam.predict(points, labels)` **once per detection box** (cheap decoder). Box prompt
is `points = [[x0,y0],[x1,y1]]` with `labels = [2, 3]` (top-left, bottom-right — SAM convention).
`predict()` returns a torch CUDA tensor `(1, N, H, W)` at input resolution; convert with
`(mask[0,0] > 0).detach().cpu().numpy()`.

**Things that will bite if forgotten:**
- `set_image` calls `.height/.width` — must pass a `PIL.Image`, NOT a numpy array. We reuse
  `image_pil` from `cv2_to_pil(image)` that tree_demo already builds.
- Decoder TRT build can fail on TRT 10.x with `IIOneHotLayer cannot be used to compute a shape
  tensor` (nanosam issue #40). Fallback: re-export with
  `python3 -m nanosam.tools.export_sam_mask_decoder_onnx --opset 17 --return-single-mask --gelu-approximate`.
- `setup.py develop --user` writes to `~/.local`, which is wiped on container `--rm`. We rely on
  `PYTHONPATH=/data/nanosam` instead.
- `OwlPredictor` boxes (`TreeDetection.box`) come out as **xyxy pixels at input resolution** —
  feed straight to SAM, no scaling needed.

## ConanAI SAM WebUI architecture (the new UI)
Forked from NVIDIA's live-vlm-webui (Apache 2.0). Reuses: aiohttp + aiortc (WebRTC) + static
HTML/JS + NVIDIA dark theme + GPU monitor + RTSP/local-file source support + multi-session
plumbing. Replaces only the inference module.

**Key swap:** `webui/vlm_service.py` (removed) → `webui/owl_sam_service.py` (new). `OwlSamService`
exposes the same interface as `VLMService` (`process_frame`, `get_current_response`, `get_metrics`,
`update_prompt`, `update_api_settings`) plus one addition: `get_last_annotation()` returning a BGR
ndarray with boxes + masks already drawn. `webui/video_processor.py` calls it inside `recv()`; when
an annotated frame exists it's swapped into the outgoing WebRTC track (with the live frame's pts /
time_base preserved) so the browser sees masks + boxes baked into the video.

**Inference cadence:** `--process-every 3` means inference fires every 3rd incoming frame, with
the `_processing_lock` guarding overlapping invocations. Effective rate is roughly
`min(webcam_fps / 3, 1 / inference_latency)` ≈ 3–7 fps on Orin NX for OWL+SAM. Between inference
cycles the *same* annotated frame keeps being emitted (so display rate ≈ inference rate while
inference is engaged). That's intentional — masks are the value-add; smooth-but-stale masks would
misalign with moving content.

**Why `vlm_service` survives as a dict key / aliased import:** server.py is ~1k LOC of lifted code
with many `session["vlm_service"]` references; renaming would multiply the diff for no behavioral
gain. video_processor.py uses `from .owl_sam_service import OwlSamService as VLMService` for the
same reason. Future refactor can rename if it ever becomes confusing.

**Repo layout (decided):** this project is one **self-contained** repo living at `~/Public/nanosam/`,
a sibling of the existing `~/Public/vlm/` project and the future `yolo/`. Each project carries its
**own** `container/` (container glue + `/data` mount), `scripts/`, and `webui/` — clone one folder,
install, run; no shared top-level infra. `~/Public/` is just the local folder that groups the
projects (not a repo). **`jetson-containers` is a shared sibling dependency** at
`~/Public/jetson-containers` (used by both nanosam and vlm) — it is NOT vendored into this repo;
`setup_host.sh` clones it and `container/run_demo.sh` locates it via `JETSON_CONTAINERS_DIR`.

**Future-merger reservation (DEFERRED, not shared-now):** the NVIDIA theme CSS variables
(`--nvidia-green`, `--bg-primary`, etc.) are kept verbatim in `webui/static/index.html`, and each
project's webui reuses the same variable names. We deliberately do **not** share a `webui/` yet —
per-project UIs diverge (SAM overlays vs VLM captions vs YOLO boxes) and a shared repo would break the
one-folder-clone goal. Only if keeping the ~90%-common framework in sync becomes painful *after a
second real spoke exists* do we extract the common core into a small library.

## Next steps after the webui is live
- **AOI logic**: per-slot ROIs, presence/count, PASS/FAIL + MISSING/MISPLACED overlay.
- **Mode selector** in the UI: OWL+SAM / OWL-only / SAM-only (click-to-segment, no text prompt).
- **Source dropdown** in the UI: image upload / video upload / webcam / RTSP.
- (Optional) SAM → VLM chain.
- ConanAI hub + shared CSS extraction once a second polished spoke exists.
- Booth polish: USB camera plumbing, fullscreen UI, lighting, pitch script.

## Wider project decisions (context, not part of this demo)

- **Factory AOI / kitting verification** (~12 product models): trained **YOLO (YOLO11 / YOLO26)** on a
  shared **part vocabulary** + per-product **recipe** (parts, counts, ROIs). SAM/MobileSAM used mainly
  for **auto-labeling** training data, not runtime.
- **Scratch on dark metal**: not YOLO — **anomaly detection** (Anomalib: EfficientAD / PatchCore) on
  good parts only; invest first in **lighting/optics** (dark-field / grazing / photometric stereo).
- **Licensing**: Ultralytics YOLO (v5/v8/v11/v26) is **AGPL-3.0** (commercial closed-source ⇒ Enterprise
  License). Apache-2.0 alternatives: YOLOX, PP-YOLOE, DAMO-YOLO. NanoOWL/NanoSAM/MobileSAM are permissive.
