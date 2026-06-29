#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Host bootstrap for the NanoOWL / NanoSAM live demo on a fresh NVIDIA Jetson.
# Target: JetPack 6.x / L4T R36.x (tested on Orin NX 16GB, JetPack 6.2 / R36.4.4).
# Run this on the Jetson HOST (not inside a container). It is safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

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
if [ ! -d "$HOME/jetson-containers" ]; then
  git clone --depth 1 https://github.com/dusty-nv/jetson-containers "$HOME/jetson-containers"
fi
bash "$HOME/jetson-containers/install.sh"

echo
echo "== DONE. Verify, then launch the container: ==============="
echo "   docker info | grep -i runtime      # expect 'nvidia' listed"
echo "   docker run --rm hello-world        # expect 'Hello from Docker!'"
echo "   jetson-containers run \$(autotag nanoowl)"
echo "   # then inside the container:  bash /data/scripts/container_setup.sh"
