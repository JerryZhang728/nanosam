#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Host bootstrap for the NanoOWL / NanoSAM live demo on a fresh NVIDIA Jetson.
# Target: JetPack 6.x / L4T R36.x (tested on Orin NX 16GB, JetPack 6.2 / R36.4.4).
# Run this on the Jetson HOST (not inside a container). It is safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root so this works no matter where it's invoked from.
REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$REPO"

echo "== Preflight: require L4T; ensure host basics ============="
# This demo runs the models INSIDE the dustynv/nanoowl container, which bundles
# CUDA + TensorRT. The HOST only needs L4T (GPU driver + DLA libs) plus Docker and
# the NVIDIA container runtime — it does NOT need CUDA or TensorRT installed natively.
if ! dpkg-query -W -f='${Version}' nvidia-l4t-core >/dev/null 2>&1; then
  echo "ERROR: 'nvidia-l4t-core' not found — this is not an NVIDIA L4T (Jetson) system." >&2
  echo "       Flash the board with NVIDIA L4T / JetPack first, then re-run this script." >&2
  echo "       (You do NOT need CUDA/TensorRT on the host — they ship inside the container —" >&2
  echo "        but the base L4T BSP must be present.)" >&2
  exit 1
fi
L4T_VER="$(dpkg-query -W -f='${Version}' nvidia-l4t-core)"
echo ">> L4T detected: nvidia-l4t-core=$L4T_VER"
L4T_TARGET="36.4"
case "$L4T_VER" in
  ${L4T_TARGET}*) echo ">> matches target R${L4T_TARGET}.x (the nanoowl image build) ✓" ;;
  *) echo ">> WARNING: host L4T ($L4T_VER) != target R${L4T_TARGET}.x — the nanoowl image is" >&2
     echo "            built for R${L4T_TARGET}.x; a version mismatch can break the driver mount." >&2
     echo "            Continuing; if inference fails, use an image tag matching your L4T." >&2 ;;
esac
# A minimal L4T flash can lack these: git/curl are used below, python3-pip is needed
# by jetson-containers' install.sh.
NEED=()
command -v git  >/dev/null 2>&1 || NEED+=(git)
command -v curl >/dev/null 2>&1 || NEED+=(curl)
command -v pip3 >/dev/null 2>&1 || NEED+=(python3-pip)
if [ "${#NEED[@]}" -gt 0 ]; then
  echo ">> installing missing host basics: ${NEED[*]}"
  sudo apt-get update
  sudo apt-get install -y "${NEED[@]}"
else
  echo ">> host basics present (git, curl, pip3) ✓"
fi

echo "==[1/5] Docker ============================================"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo usermod -aG docker "$USER"
  echo ">> Added '$USER' to the docker group."
  echo ">> You must log out / back in (or run 'newgrp docker') for this to take effect."
else
  echo ">> docker already installed."
fi

echo "==[2/5] NVIDIA container toolkit =========================="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
else
  echo ">> nvidia-container-toolkit already installed."
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==[3/5] Enable L4T apt repo ==============================="
# Needed so we can install the DLA compiler lib below. The repo lines ship commented out.
L4T_SRC=/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
if [ -f "$L4T_SRC" ]; then
  sudo sed -i 's|^#deb \(https://repo.download.nvidia.com/jetson/common\)|deb \1|' "$L4T_SRC"
  sudo sed -i 's|^#deb \(https://repo.download.nvidia.com/jetson/t234\)|deb \1|'  "$L4T_SRC"
  sudo apt-get update
else
  echo ">> WARNING: $L4T_SRC not found — L4T repo may already be configured elsewhere."
fi

echo "==[4/5] Fix missing libnvdla_compiler.so ================="
# GOTCHA: minimized JetPack flashes ship libnvdla_runtime.so but NOT libnvdla_compiler.so.
# Without it, TensorRT inside the container fails:
#   ImportError: libnvdla_compiler.so: cannot open shared object file
# Install the version that MATCHES nvidia-l4t-core (mixing L4T versions breaks things).
DLA=/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so
if [ ! -f "$DLA" ]; then
  CORE_VER=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core)
  echo ">> nvidia-l4t-core=$CORE_VER — installing matching nvidia-l4t-dla-compiler"
  sudo apt-get install -y "nvidia-l4t-dla-compiler=${CORE_VER}"
else
  echo ">> libnvdla_compiler.so already present."
fi
# Confirm it's in the CSV mount list (so the runtime injects it into containers)
grep -q libnvdla_compiler.so /etc/nvidia-container-runtime/host-files-for-container.d/drivers.csv \
  && echo ">> libnvdla_compiler.so is in drivers.csv (will be mounted into containers)." \
  || echo ">> NOTE: not in drivers.csv — add a line if the container still can't find it."

echo "==[5/5] jetson-containers ================================="
# Clone location MUST match run_demo.sh's default + the README staging path
# (~/Public/jetson-containers), or the demo won't find the container / data mount.
JC="${JETSON_CONTAINERS_DIR:-$HOME/Public/jetson-containers}"
if [ ! -d "$JC" ]; then
  mkdir -p "$(dirname "$JC")"
  git clone --depth 1 https://github.com/dusty-nv/jetson-containers "$JC"
fi
bash "$JC/install.sh"

echo
echo "== Smoke test: Docker + NVIDIA runtime ===================="
DOCKER_OK=0
if docker info >/dev/null 2>&1; then
  DOCKER_OK=1
  docker info 2>/dev/null | grep -qi nvidia \
    && echo ">> NVIDIA container runtime registered ✓" \
    || echo ">> WARNING: nvidia runtime not visible in 'docker info' — recheck step 2." >&2
  docker run --rm hello-world >/dev/null 2>&1 \
    && echo ">> 'docker run' works ✓" \
    || echo ">> WARNING: 'docker run hello-world' failed — recheck the Docker install." >&2
else
  echo ">> Docker not usable from THIS shell yet (docker-group change pending)."
fi

echo
echo "== Install the 'sam-demo' launcher ======================="
sudo chmod +x "$REPO/container/run_demo.sh"
sudo ln -sfn "$REPO/container/run_demo.sh" /usr/local/bin/sam-demo
echo ">> installed: sam-demo -> $REPO/container/run_demo.sh"

echo
echo "===================== SETUP COMPLETE ====================="
if [ "$DOCKER_OK" -ne 1 ]; then
  echo "  1) Activate the docker group (one time):   newgrp docker"
  echo "  2) Run the demo:                           sam-demo"
else
  echo "  Run the demo:                              sam-demo"
fi
echo "  Then open:  https://<this-jetson-ip>:7860   (accept the self-signed cert)"
echo "              or  https://localhost:7860  on the Jetson itself"
echo "  First 'sam-demo' pulls the ~GB image + builds TRT engines — give it a few minutes."
echo "=========================================================="
