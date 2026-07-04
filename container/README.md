# container/ — Docker / jetson-containers integration for this project

This project runs inside a **jetson-containers** `nanoowl` container. This folder holds the
container-side glue; the actual image build is provided by jetson-containers (installed by
`../setup_host.sh`).

## Contents
- `run_demo.sh` — host-side launcher. Enters the `nanoowl` container (via jetson-containers) and
  runs `container_setup.sh`, which builds the TRT engines (if missing) and serves the web UI on
  **:7860** (HTTPS, self-signed). Override the checkout location with
  `JETSON_CONTAINERS_DIR=/path/to/jetson-containers ./run_demo.sh`.
- `data/` — the **`/data` bind mount**. Everything the container persists (built `*.engine` files,
  the NanoSAM checkout, `test.mp4`, sample images) lives here. Contents are git-ignored (large /
  generated); only `.gitkeep` is tracked so the dir survives a clone.

## The staging contract (why the repo layout ≠ the runtime layout)
The container runs `--rm`, so **only `/data` persists**. At runtime everything is referenced by
absolute container paths — `/data/scripts`, `/data/webui`, `/data/nanosam`. Those are populated by
**staging**: before launching, the repo's `../scripts/` and `../webui/` are copied into the mounted
`data/` dir so they appear as `/data/scripts` and `/data/webui` inside the container. See the repo
`README.md` "Quick start" for the exact copy commands.

> Because of this staging step you can reorganize the repo freely — the hardcoded `/data/...` paths
> keep working as long as staging still lands `scripts/` → `/data/scripts` and `webui/` → `/data/webui`.
