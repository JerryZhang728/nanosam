#!/usr/bin/env bash
# =============================================================================
# enable_host_logging.sh  --  PERSONAL host diagnostics for a Jetson.
# NOT part of the demo. Run it yourself when you want the box to keep logs that
# survive a reboot / crash, so you can find out WHY it died. Safe to re-run.
#
#   bash scripts/enable_host_logging.sh
#
# Enables:
#   1. Persistent journald  -> crash logs survive a reboot   (journalctl -b -1 -e)
#   2. tegrastats logger    -> temp/mem/GPU/power sampled every 3s to a file, so
#      even a HARD power-cut leaves the run-up (last line = moment before freeze).
#
# Read afterwards:
#   previous boot : journalctl -b -1 -e
#   telemetry     : tail -f /var/log/jetson-tegrastats.log
# =============================================================================
set -uo pipefail

echo "== [1/2] Persistent journald ============================="
sudo mkdir -p /var/log/journal
JC=/etc/systemd/journald.conf
if grep -qE '^#?Storage=' "$JC"; then sudo sed -i -E 's/^#?Storage=.*/Storage=persistent/' "$JC"; else echo 'Storage=persistent' | sudo tee -a "$JC" >/dev/null; fi
if grep -qE '^#?SystemMaxUse=' "$JC"; then sudo sed -i -E 's/^#?SystemMaxUse=.*/SystemMaxUse=500M/' "$JC"; else echo 'SystemMaxUse=500M' | sudo tee -a "$JC" >/dev/null; fi
sudo systemd-tmpfiles --create --prefix /var/log/journal 2>/dev/null || true
sudo systemctl restart systemd-journald
echo ">> journald persistent (cap 500M).  Read a crash with:  journalctl -b -1 -e"

echo
echo "== [2/2] tegrastats logger service ======================="
TS="$(command -v tegrastats || echo /usr/bin/tegrastats)"
if [ ! -x "$TS" ]; then echo ">> WARNING: tegrastats not found at '$TS' - skipping telemetry logger."; exit 0; fi

# Helper that timestamps each tegrastats line (kept out of the unit file to avoid
# systemd %-escaping). $TS is baked in now; $(date)/$line are deferred to runtime.
sudo tee /usr/local/bin/jetson-tegrastats-logger >/dev/null <<EOF
#!/bin/bash
exec $TS --interval 3000 | while IFS= read -r line; do
  printf '%s %s\n' "\$(date '+%F %T')" "\$line"
done >> /var/log/jetson-tegrastats.log
EOF
sudo chmod +x /usr/local/bin/jetson-tegrastats-logger

sudo tee /etc/systemd/system/tegrastats-logger.service >/dev/null <<'EOF'
[Unit]
Description=tegrastats telemetry logger (temp/mem/GPU/power -> /var/log/jetson-tegrastats.log)
After=multi-user.target

[Service]
ExecStart=/usr/local/bin/jetson-tegrastats-logger
Restart=always
RestartSec=5
Nice=10

[Install]
WantedBy=multi-user.target
EOF

# Rotate so it can never fill the disk.
sudo tee /etc/logrotate.d/jetson-tegrastats >/dev/null <<'EOF'
/var/log/jetson-tegrastats.log {
    daily
    rotate 7
    size 50M
    compress
    missingok
    notifempty
    copytruncate
}
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now tegrastats-logger.service
sleep 3
echo ">> tegrastats-logger: $(systemctl is-active tegrastats-logger.service)"
echo ">> latest samples:"
tail -n 3 /var/log/jetson-tegrastats.log 2>/dev/null || echo "   (no samples yet)"

echo
echo "===================== DONE ====================="
echo "  crash logs : journalctl -b -1 -e"
echo "  telemetry  : tail -f /var/log/jetson-tegrastats.log"
echo "================================================"
