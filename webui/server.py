# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebRTC Live VLM WebUI Server
Main server that handles WebRTC connections and serves the web interface
"""

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import uuid
from collections import defaultdict

import aiohttp
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaRelay

from .owl_sam_service import OwlSamService
from .video_processor import VideoProcessorTrack
from .gpu_monitor import create_monitor
from .rtsp_track import RTSPVideoTrack
from .local_file_track import LocalFileVideoTrack

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global objects
relay = MediaRelay()
pcs = set()
vlm_service = None  # Kept for backwards compat; default session uses sessions["default"]
websockets = set()  # Track active WebSocket connections (all)
gpu_monitor = None  # GPU monitoring instance
gpu_monitor_task = None  # Background task for GPU monitoring
rtsp_tracks = {}  # Track active RTSP streams {session_id: (rtsp_track, processor_track)}

# Multi-session state (0.4.0)
default_vlm_config = {}  # Set at startup; used to create new sessions
sessions = {}  # session_id -> {"vlm_service": VLMService}
session_websockets = defaultdict(set)  # session_id -> set of ws
ws_to_session = {}  # ws -> session_id


def get_or_create_session(session_id: str):
    """Get or create per-session state (inference service). Thread-safe for aiohttp.
    Note: the dict key "vlm_service" is kept so the rest of server.py (lifted from
    vlm2) keeps working unchanged — it now holds an OwlSamService instance.

    ALL OwlSamService constructor args must be passed through default_vlm_config so
    that browser-side sessions (each one is a new session_id) inherit the same
    engine/model choices that main() set up — otherwise we silently fall back to
    constructor defaults and get model/engine geometry mismatches."""
    if session_id not in sessions:
        # SHARED model backend: every session reuses the ONE global OwlSamService, so the OWL+SAM
        # TensorRT engines load exactly once. Previously each session_id built its OWN OwlSamService,
        # and each lazily loaded ~1GB of OWL+SAM onto the GPU that was never freed — so reloading /
        # reconnecting the page a handful of times stacked the copies (6 reloads -> ~6GB torch ->
        # ~14.7GB total -> near OOM). This is a single-user booth demo, so one shared prompt/mode is
        # fine; the _processing_lock inside the service already serializes overlapping inference.
        global vlm_service
        if vlm_service is None:
            cfg = default_vlm_config
            vlm_service = OwlSamService(
                owl_engine=cfg.get("owl_engine", "/data/owl_image_encoder_patch32.engine"),
                sam_image_encoder_engine=cfg.get(
                    "sam_image_encoder_engine", "/data/nanosam/data/resnet18_image_encoder.engine"
                ),
                sam_mask_decoder_engine=cfg.get(
                    "sam_mask_decoder_engine", "/data/nanosam/data/mobile_sam_mask_decoder.engine"
                ),
                prompt=cfg.get("prompt", "[a person]"),
                mask_alpha=cfg.get("mask_alpha", 0.5),
                owl_threshold=cfg.get("owl_threshold", 0.1),
                owl_model_name=cfg.get("owl_model_name", "google/owlvit-base-patch32"),
            )
        sessions[session_id] = {
            "vlm_service": vlm_service,       # shared — do NOT construct a new one per session
            "show_request_payload": False,
            "show_response_payload": False,
        }
        logger.info(f"Created session {session_id} (shared model backend)")
    return sessions[session_id]


def send_to_session(session_id: str, message: str):
    """Send a message only to WebSocket clients in this session."""
    for ws in session_websockets.get(session_id, set()):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")


def get_session_callback(session_id: str):
    """Return a text_callback that sends VLM results only to this session."""

    def callback(text: str, metrics: dict):
        out = {"type": "vlm_response", "text": text, "metrics": metrics}
        session = sessions.get(session_id)
        if session and session.get("vlm_service"):
            svc = session["vlm_service"]
            if session.get("show_request_payload"):
                payload = svc.get_last_request_payload()
                if payload is not None:
                    out["request_payload"] = payload
            if session.get("show_response_payload"):
                payload = svc.get_last_response_payload()
                if payload is not None:
                    try:
                        out["response_payload"] = json.loads(json.dumps(payload, default=str))
                    except (TypeError, ValueError):
                        out["response_payload"] = payload
        send_to_session(session_id, json.dumps(out))

    return callback


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available for binding"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def find_process_using_port(port):
    """Find what process is using a port (Linux/Unix only)"""
    try:
        # Try lsof first (more reliable)
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            # Get process name
            name_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True, timeout=2
            )
            if name_result.returncode == 0:
                return f"PID {pid} ({name_result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available, try netstat
        try:
            result = subprocess.run(
                ["netstat", "-tulpn"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        return parts[-1]  # PID/Program name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return "unknown process"


def find_available_port(start_port=8080, max_attempts=10):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


async def detect_local_service_and_model():
    """
    Auto-detect available local VLM services and select a model
    Returns: (api_base, model_name) or (None, None) if no service found
    """
    services = [
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:30000/v1", "SGLang"),
    ]

    for api_base, service_name in services:
        try:
            # Try to connect to the service
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{api_base}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Prefer vision models
                            vision_keywords = ["vision", "llava", "llama-3.2", "gemini"]
                            for model in models:
                                model_id = model.get("id", "")
                                if any(keyword in model_id.lower() for keyword in vision_keywords):
                                    logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                                    logger.info(f"   Selected model: {model_id}")
                                    return (api_base, model_id)

                            # If no vision model found, use the first one
                            model_id = models[0].get("id", "")
                            logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                            logger.info(
                                f"   Selected model: {model_id} (vision model preferred but not found)"
                            )
                            return (api_base, model_id)
        except Exception as e:
            logger.debug(f"Service {service_name} not available at {api_base}: {e}")
            continue

    return (None, None)


async def index(request):
    """Serve the main HTML page"""
    content = open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def models(request):
    """Three diagnostic modes exposed as "models" so the existing UI dropdown drives them.
    OWL-only is listed first because it's the default and most reliable mode."""
    options = [
        {"id": "nanoowl",                     "name": "NanoOWL only (boxes)"},
        {"id": "nanoowl+bytetrack",           "name": "NanoOWL + ByteTrack (tracked IDs)"},
        {"id": "nanoowl+nanosam",             "name": "NanoOWL + NanoSAM (box → mask)"},
        {"id": "nanoowl+bytetrack+nanosam",   "name": "NanoOWL + ByteTrack + NanoSAM (tracked masks)"},
    ]
    current = "nanoowl+nanosam"
    if sessions.get("default"):
        current = getattr(sessions["default"]["vlm_service"], "model", current)
    for o in options:
        o["current"] = (o["id"] == current)
    return web.Response(
        content_type="application/json",
        text=json.dumps({"models": options}),
    )


async def detect_services(request):
    """Detect available local VLM services"""
    services = [
        {"name": "Ollama", "url": "http://localhost:11434/v1", "port": 11434, "path": "/api/tags"},
        {"name": "vLLM", "url": "http://localhost:8000/v1", "port": 8000, "path": "/v1/models"},
        {"name": "SGLang", "url": "http://localhost:30000/v1", "port": 30000, "path": "/v1/models"},
    ]

    detected = []

    async def check_service(service):
        """Check if a service is running by probing its endpoint"""
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://localhost:{service['port']}{service['path']}"
                async with session.get(url) as response:
                    if response.status in [200, 404]:  # 404 is ok, means server is running
                        logger.info(f"Detected {service['name']} at {service['url']}")
                        return service
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        return None

    # Check all services concurrently
    results = await asyncio.gather(*[check_service(s) for s in services])
    detected = [s for s in results if s is not None]

    # Default to NVIDIA API Catalog if no local services found
    if not detected:
        detected.append(
            {
                "name": "NVIDIA API Catalog",
                "url": "https://integrate.api.nvidia.com/v1",
                "port": None,
                "path": None,
                "requires_key": True,
            }
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"detected": detected, "default": detected[0] if detected else None}),
    )


async def websocket_handler(request):
    """Handle WebSocket connections for text updates. Supports ?session_id= for multi-session."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session ID from query or generate new (client should send same id in /offer)
    session_id = request.query.get("session_id", "").strip() or str(uuid.uuid4())
    ws_to_session[ws] = session_id
    session_websockets[session_id].add(ws)
    websockets.add(ws)
    logger.info(
        f"WebSocket client connected. session_id={session_id}, total clients: {len(websockets)}"
    )

    session = get_or_create_session(session_id)
    svc = session["vlm_service"]

    try:
        # Send initial message with current server configuration (include session_id if we generated it)
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to server",
                "status": "Ready",
                "session_id": session_id,
            }
        )

        # Send current server configuration for this session
        from .video_processor import VideoProcessorTrack as _VPT

        await ws.send_json(
            {
                "type": "server_config",
                "model": svc.model,
                "api_base": svc.api_base,
                "prompt": svc.prompt,
                "process_every": _VPT.process_every_n_frames,
                "session_id": session_id,
            }
        )

        # Keep connection alive and handle incoming messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Re-resolve session in case it was recreated
                    svc = get_or_create_session(session_id)["vlm_service"]

                    if data.get("type") == "update_prompt":
                        new_prompt = data.get("prompt", "").strip()
                        max_tokens = data.get("max_tokens")
                        if new_prompt and svc:
                            svc.update_prompt(new_prompt, max_tokens)
                            logger.info(
                                f"[{session_id}] Prompt updated: {new_prompt}, max_tokens: {max_tokens}"
                            )

                            await ws.send_json(
                                {
                                    "type": "prompt_updated",
                                    "prompt": new_prompt,
                                    "max_tokens": max_tokens,
                                }
                            )

                    elif data.get("type") == "update_model":
                        new_model = data.get("model", "").strip()
                        api_base = data.get("api_base", "").strip()
                        api_key = data.get("api_key", "").strip()

                        if new_model and svc:
                            svc.model = new_model
                            if api_base:
                                svc.update_api_settings(api_base, api_key if api_key else None)
                                logger.info(
                                    f"[{session_id}] Model updated: {new_model}, API: {api_base}"
                                )
                            else:
                                logger.info(f"[{session_id}] Model updated: {new_model}")

                            await ws.send_json(
                                {
                                    "type": "model_updated",
                                    "model": new_model,
                                    "api_base": svc.api_base,
                                }
                            )

                    elif data.get("type") == "update_processing":
                        process_every = data.get("process_every", 30)
                        try:
                            process_every = int(process_every)
                            if 1 <= process_every <= 3600:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.process_every_n_frames
                                VideoProcessorTrack.process_every_n_frames = process_every
                                logger.info(
                                    f"[{session_id}] Processing interval updated: {old_value} → {process_every} frames"
                                )

                                await ws.send_json(
                                    {"type": "processing_updated", "process_every": process_every}
                                )
                            else:
                                logger.warning(
                                    f"Processing interval out of range (1-3600): {process_every}"
                                )
                        except ValueError:
                            logger.error(f"Invalid processing interval: {process_every}")

                    elif data.get("type") == "set_debug":
                        session_data = get_or_create_session(session_id)
                        if "show_request_payload" in data:
                            session_data["show_request_payload"] = bool(
                                data["show_request_payload"]
                            )
                        if "show_response_payload" in data:
                            session_data["show_response_payload"] = bool(
                                data["show_response_payload"]
                            )
                        logger.debug(
                            f"[{session_id}] Debug: request_payload="
                            f"{session_data.get('show_request_payload')}, response_payload="
                            f"{session_data.get('show_response_payload')}"
                        )

                    elif data.get("type") == "update_max_latency":
                        max_latency = data.get("max_latency", 0.0)
                        try:
                            max_latency = float(max_latency)
                            if 0 <= max_latency <= 10.0:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.max_frame_latency
                                VideoProcessorTrack.max_frame_latency = max_latency
                                status = "disabled" if max_latency == 0 else f"{max_latency:.1f}s"
                                old_status = "disabled" if old_value == 0 else f"{old_value:.1f}s"
                                logger.info(
                                    f"[{session_id}] Max frame latency updated: {old_status} → {status}"
                                )

                                await ws.send_json(
                                    {"type": "max_latency_updated", "max_latency": max_latency}
                                )
                            else:
                                logger.warning(f"Max latency out of range (0-10.0): {max_latency}")
                        except ValueError:
                            logger.error(f"Invalid max latency value: {max_latency}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        session_websockets[session_id].discard(ws)
        ws_to_session.pop(ws, None)
        websockets.discard(ws)
        logger.info(
            f"WebSocket client disconnected. session_id={session_id}, total clients: {len(websockets)}"
        )

    return ws


def broadcast_text_update(text: str, metrics: dict):
    """Broadcast text update and metrics to all connected WebSocket clients"""
    if not websockets:
        return

    message = json.dumps({"type": "vlm_response", "text": text, "metrics": metrics})

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            # Use asyncio to send without blocking
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


def broadcast_gpu_stats(stats: dict):
    """Broadcast GPU stats to all connected WebSocket clients"""
    if not websockets:
        return

    message = json.dumps({"type": "gpu_stats", "stats": stats})

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending GPU stats to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


async def gpu_monitor_loop():
    """Background task to periodically collect and broadcast GPU stats"""
    global gpu_monitor

    if not gpu_monitor:
        logger.warning("GPU monitor not initialized, skipping monitoring")
        return

    logger.info("GPU monitoring loop started")

    try:
        while True:
            # Get current stats
            stats = gpu_monitor.get_stats()

            # Update history with current stats
            gpu_monitor.update_history(stats)

            # Add history to stats
            stats["history"] = gpu_monitor.get_history()

            # Broadcast to all connected clients
            broadcast_gpu_stats(stats)

            # Update every 0.25 seconds for detailed GPU monitoring
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        logger.info("GPU monitoring loop cancelled")
    except Exception as e:
        logger.error(f"Error in GPU monitoring loop: {e}")



async def browse_files(request):
    """
    Filesystem browse endpoint for the Browse... button in the UI.

    GET /api/browse?path=/abs/path&mode=image|video

    Returns: { "path": str, "parent": str, "entries": [{name, is_dir, size}] }

    Filters files by mode (only show matching image/video extensions). Folders
    always shown. Symlinks are followed. Permission errors return 403.
    """
    from pathlib import Path
    raw_path = request.query.get("path", "/data")
    mode = request.query.get("mode", "image").lower()

    if mode == "image":
        allowed = LocalFileVideoTrack.IMAGE_EXTS
    elif mode == "video":
        allowed = LocalFileVideoTrack.VIDEO_EXTS
    else:
        allowed = LocalFileVideoTrack.IMAGE_EXTS | LocalFileVideoTrack.VIDEO_EXTS

    p = Path(raw_path).expanduser()
    try:
        p = p.resolve()
    except OSError as e:
        return web.json_response({"error": f"Cannot resolve path: {e}"}, status=400)

    if not p.exists():
        return web.json_response({"error": f"Path does not exist: {p}"}, status=404)
    if not p.is_dir():
        # If user typed a file path, browse its parent directory instead
        p = p.parent

    try:
        children = list(p.iterdir())
    except PermissionError:
        return web.json_response({"error": f"Permission denied: {p}"}, status=403)
    except OSError as e:
        return web.json_response({"error": str(e)}, status=500)

    entries = []
    for child in sorted(children, key=lambda c: (not c.is_dir(), c.name.lower())):
        # Hide dotfiles by default
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            if is_dir:
                entries.append({"name": child.name, "is_dir": True})
            elif child.suffix.lower() in allowed:
                try:
                    sz = child.stat().st_size
                except OSError:
                    sz = None
                entries.append({"name": child.name, "is_dir": False, "size": sz})
        except OSError:
            continue

    parent = str(p.parent) if p.parent != p else str(p)
    return web.json_response({"path": str(p), "parent": parent, "entries": entries})


async def offer(request):
    """Handle WebRTC offer from client.

    Supports four source modes, dispatched by which optional parameter is
    present in the offer body:

      * rtsp_url            -> RTSP IP camera (server-side decode)
      * local_image_path    -> single image file OR folder of images (slideshow).
                               Auto-detected by path.
      * local_video_path    -> local video file (mp4/avi/mkv/mov/...).
      * (none of the above) -> webcam (browser captures, server processes).

    Also accepts optional analysis-cadence parameters that override
    VideoProcessorTrack defaults for this session:

      * analysis_mode               "frames" | "seconds" | "scene_change"
      * process_every_n_frames      int, used when mode == "frames"
      * analysis_interval_seconds   float, used when mode == "seconds"
      * scene_change_threshold      float (0-255), used when mode == "scene_change"
      * scene_change_min_interval   float seconds, debounce for scene_change

    Plus local-file behavior:

      * image_fps                       float, output FPS for still images
      * slideshow_seconds_per_image     float, hold time per slideshow image
      * slideshow_fps                   float, output FPS during slideshow
      * video_loop                      bool, restart at EOF (default True)
    """
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    rtsp_url = params.get("rtsp_url")  # Optional RTSP URL for IP camera mode
    local_image_path = params.get("local_image_path")  # Image file or folder
    local_video_path = params.get("local_video_path")  # Video file
    session_id = params.get("session_id", "default")

    # ---- Apply optional cadence overrides --------------------------------
    # These are class-level on VideoProcessorTrack (same as
    # process_every_n_frames already is), so they affect all newly-created
    # processor tracks. Sufficient for typical single-user workflow.
    if "analysis_mode" in params:
        VideoProcessorTrack.analysis_mode = str(params["analysis_mode"])
    if "process_every_n_frames" in params:
        try:
            VideoProcessorTrack.process_every_n_frames = max(1, int(params["process_every_n_frames"]))
        except (TypeError, ValueError):
            pass
    if "analysis_interval_seconds" in params:
        try:
            VideoProcessorTrack.analysis_interval_seconds = max(0.1, float(params["analysis_interval_seconds"]))
        except (TypeError, ValueError):
            pass
    if "scene_change_threshold" in params:
        try:
            VideoProcessorTrack.scene_change_threshold = max(0.0, float(params["scene_change_threshold"]))
        except (TypeError, ValueError):
            pass
    if "scene_change_min_interval" in params:
        try:
            VideoProcessorTrack.scene_change_min_interval = max(0.0, float(params["scene_change_min_interval"]))
        except (TypeError, ValueError):
            pass
    logger.info(
        f"[{session_id}] Analysis cadence: mode={VideoProcessorTrack.analysis_mode}, "
        f"frames_n={VideoProcessorTrack.process_every_n_frames}, "
        f"seconds={VideoProcessorTrack.analysis_interval_seconds}, "
        f"scene_threshold={VideoProcessorTrack.scene_change_threshold}, "
        f"scene_min_interval={VideoProcessorTrack.scene_change_min_interval}"
    )

    session = get_or_create_session(session_id)
    session_vlm = session["vlm_service"]
    session_callback = get_session_callback(session_id)

    # Create RTCPeerConnection with STUN servers for Docker/NAT compatibility
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    # Store server-side source track for cleanup (RTSP or local file).
    # Variable name kept as rtsp_cleanup_track for backward-compat with
    # existing close handler; the actual track may be RTSPVideoTrack OR
    # LocalFileVideoTrack -- both have the same stop() interface.
    rtsp_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            # Clean up server-side source track if exists
            if rtsp_cleanup_track:
                rtsp_cleanup_track.stop()
                logger.info("Source track stopped on connection close")
            await pc.close()
            pcs.discard(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.error("ICE connection failed - check firewall/NAT settings")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state: {pc.iceGatheringState}")

    # If RTSP URL provided, create RTSP track instead of waiting for browser track
    if rtsp_url:
        logger.info(f"[{session_id}] Creating RTSP track for: {rtsp_url}")
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
            rtsp_cleanup_track = rtsp_track  # Store for cleanup

            # Wait for initial connection to get stream info
            await asyncio.sleep(0.5)

            # Wrap RTSP track with relay first (same pattern as webcam).
            # buffered=False: serve only the latest frame, DROP the backlog. With
            # the default buffered=True, aiortc queues every source frame per
            # subscriber unboundedly; when the encoder is slower than the source
            # FPS the queue grows forever (~6MB/raw frame) and OOMs the Jetson.
            relayed_rtsp = relay.subscribe(rtsp_track, buffered=False)

            processor_track = VideoProcessorTrack(
                relayed_rtsp, session_vlm, text_callback=session_callback
            )

            # Add processor directly to peer connection
            pc.addTrack(processor_track)
            logger.info("Added RTSP processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )
    elif local_image_path or local_video_path:
        # ----------------------------------------------------------------
        # Local file mode: image, image folder (slideshow), or video.
        # Mirrors the RTSP branch. Source path comes from whichever tab
        # the user picked in the UI; server treats them the same since
        # LocalFileVideoTrack auto-detects from the path.
        # ----------------------------------------------------------------
        source_path = local_image_path or local_video_path
        which_tab = "image" if local_image_path else "video"
        logger.info(
            f"[{session_id}] Creating local file track ({which_tab} tab): {source_path}"
        )
        try:
            local_track = LocalFileVideoTrack(
                source_path,
                image_fps=float(params.get("image_fps", 1.0)),
                slideshow_seconds_per_image=float(
                    params.get("slideshow_seconds_per_image", 5.0)
                ),
                slideshow_fps=float(params.get("slideshow_fps", 1.0)),
                loop=bool(params.get("video_loop", True)),
            )
            rtsp_cleanup_track = local_track  # Reuse cleanup hook

            # Quick yield so any initial async work settles before SDP.
            await asyncio.sleep(0.1)

            relayed = relay.subscribe(local_track, buffered=False)  # drop backlog, latest-frame only (see RTSP note)
            processor_track = VideoProcessorTrack(
                relayed, session_vlm, text_callback=session_callback
            )
            pc.addTrack(processor_track)

            stats = local_track.get_stats()
            logger.info(
                f"Local file track ready: mode={stats['mode']}, path={stats['path']}"
            )

        except FileNotFoundError as e:
            logger.error(f"Local file not found: {e}")
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"error": f"File not found: {source_path}"}),
            )
        except ValueError as e:
            logger.error(f"Local file invalid: {e}")
            return web.Response(
                status=415,
                content_type="application/json",
                text=json.dumps({"error": str(e)}),
            )
        except Exception as e:
            logger.error(f"Failed to create local file track: {e}", exc_info=True)
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to open source: {str(e)}"}),
            )
    else:
        # Webcam mode: wait for browser to send track
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")

            if track.kind == "video":
                # Create processor track with this session's VLM and session-scoped callback
                processor_track = VideoProcessorTrack(
                    relay.subscribe(track, buffered=False),  # drop backlog, latest-frame only (see RTSP note)
                    session_vlm, text_callback=session_callback
                )

                # Add processed track back to connection
                pc.addTrack(processor_track)
                logger.info("Added processed video track back to peer connection")

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")

    # Handle offer
    await pc.setRemoteDescription(offer_sdp)

    # Create answer - this must happen after tracks are added
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"Created answer with {len(pc.getTransceivers())} transceivers")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def rtsp_start(request):
    """
    Start RTSP stream processing.

    Accepts RTSP URL and creates a video processing pipeline.

    POST /api/rtsp/start
    Body: {"rtsp_url": "rtsp://...", "session_id": "optional-id"}
    """
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url")
        session_id = data.get("session_id", "default")

        if not rtsp_url:
            logger.warning("RTSP start request missing rtsp_url")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Missing rtsp_url parameter"}),
            )

        # Check if session already exists
        if session_id in rtsp_tracks:
            logger.warning(f"RTSP session {session_id} already exists, stopping it first")
            await _stop_rtsp_session(session_id)

        logger.info(f"Starting RTSP stream for session {session_id}")

        # Create RTSP video track
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )

        # Create processor track with this session's VLM and session-scoped callback
        session = get_or_create_session(session_id)
        session_vlm = session["vlm_service"]
        session_callback = get_session_callback(session_id)
        processor_track = VideoProcessorTrack(
            rtsp_track, session_vlm, text_callback=session_callback
        )

        # Start background task to consume frames
        async def consume_frames():
            """Background task to continuously pull frames from processor track"""
            try:
                while not rtsp_track._stopped:
                    try:
                        _ = await processor_track.recv()
                        # Frame is processed, just discard it (VLM analysis happens in recv())
                    except StopAsyncIteration:
                        logger.info(f"RTSP stream {session_id} ended")
                        break
                    except Exception as e:
                        logger.error(f"Error consuming RTSP frame for {session_id}: {e}")
                        break
            finally:
                logger.info(f"Frame consumption stopped for {session_id}")

        frame_task = asyncio.create_task(consume_frames())

        # Store reference with frame task
        rtsp_tracks[session_id] = (rtsp_track, processor_track, frame_task)

        # Get stream stats
        stats = rtsp_track.get_stats()

        logger.info(
            f"RTSP stream started: {session_id} - {stats.get('codec')} "
            f"{stats.get('width')}x{stats.get('height')}"
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "started", "session_id": session_id, "stream_info": stats}),
        )

    except Exception as e:
        logger.error(f"Error starting RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_stop(request):
    """
    Stop RTSP stream processing.

    POST /api/rtsp/stop
    Body: {"session_id": "optional-id"}
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")

        await _stop_rtsp_session(session_id)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "stopped", "session_id": session_id}),
        )

    except Exception as e:
        logger.error(f"Error stopping RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_status(request):
    """
    Get status of all RTSP streams.

    GET /api/rtsp/status
    """
    try:
        status_list = []

        for session_id, (rtsp_track, processor_track, frame_task) in rtsp_tracks.items():
            stats = rtsp_track.get_stats()
            status_list.append(
                {
                    "session_id": session_id,
                    "connected": stats.get("connected"),
                    "frames_received": stats.get("frames_received"),
                    "stream_info": {
                        "codec": stats.get("codec"),
                        "width": stats.get("width"),
                        "height": stats.get("height"),
                        "fps": stats.get("fps"),
                    },
                }
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"active_streams": len(rtsp_tracks), "streams": status_list}),
        )

    except Exception as e:
        logger.error(f"Error getting RTSP status: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def _stop_rtsp_session(session_id: str):
    """Helper function to stop an RTSP session"""
    if session_id in rtsp_tracks:
        rtsp_track, processor_track, frame_task = rtsp_tracks[session_id]

        # Cancel frame consumption task
        if frame_task and not frame_task.done():
            frame_task.cancel()
            try:
                await frame_task
            except asyncio.CancelledError:
                pass

        # Stop tracks
        try:
            processor_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping processor track: {e}")

        try:
            rtsp_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping RTSP track: {e}")

        # Remove from tracking
        del rtsp_tracks[session_id]
        logger.info(f"RTSP stream stopped: {session_id}")
    else:
        logger.warning(f"RTSP session {session_id} not found")


async def on_startup(app):
    """Initialize resources on server startup"""
    global gpu_monitor, gpu_monitor_task

    # Initialize GPU monitor
    try:
        gpu_monitor = create_monitor()
        logger.info("GPU monitor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitor: {e}")
        gpu_monitor = None

    # Start GPU monitoring background task
    if gpu_monitor:
        gpu_monitor_task = asyncio.create_task(gpu_monitor_loop())
        logger.info("GPU monitoring task started")


async def on_shutdown(app):
    """Cleanup on server shutdown"""
    global gpu_monitor, gpu_monitor_task

    logger.info("Shutting down server...")

    # Stop GPU monitoring task
    if gpu_monitor_task:
        gpu_monitor_task.cancel()
        try:
            await gpu_monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("GPU monitoring task stopped")

    # Cleanup GPU monitor
    if gpu_monitor:
        gpu_monitor.cleanup()
        logger.info("GPU monitor cleaned up")

    # Close all websockets and clear session state
    for ws in list(websockets):
        await ws.close()
    websockets.clear()
    session_websockets.clear()
    ws_to_session.clear()

    # Close all RTSP streams
    for session_id in list(rtsp_tracks.keys()):
        await _stop_rtsp_session(session_id)
    logger.info("RTSP streams closed")

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    logger.info("Cleanup complete")


async def create_app(test_mode=False):
    """
    Create and configure the aiohttp web application.

    Args:
        test_mode: If True, skip GPU monitoring and use test configuration

    Returns:
        Configured web.Application instance
    """
    # Create web application
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/models", models)
    app.router.add_get("/detect-services", detect_services)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/browse", browse_files)
    app.router.add_post("/offer", offer)

    # RTSP endpoints
    app.router.add_post("/api/rtsp/start", rtsp_start)
    app.router.add_post("/api/rtsp/stop", rtsp_stop)
    app.router.add_get("/api/rtsp/status", rtsp_status)

    # Serve static files (images, etc.)
    # Always serve from static/images within the package (works for both pip and dev installs)
    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images_dir = os.path.abspath(images_dir)

    if os.path.exists(images_dir):
        app.router.add_static("/images", images_dir, name="images")
        logger.info(f"Serving static files from: {images_dir}")
    else:
        logger.warning(f"⚠️  Static images directory not found: {images_dir}")

    # Serve favicon files
    favicon_dir = os.path.join(os.path.dirname(__file__), "static", "favicon")
    favicon_dir = os.path.abspath(favicon_dir)

    if os.path.exists(favicon_dir):
        app.router.add_static("/favicon", favicon_dir, name="favicon")
        logger.info(f"Serving favicon files from: {favicon_dir}")
    else:
        logger.warning(f"⚠️  Favicon directory not found: {favicon_dir}")

    # Serve vendored JS libs (lucide/marked/dompurify) locally so the UI works with NO
    # internet — the Jetson can't reach unpkg/jsdelivr, and those were synchronous <script>
    # tags that otherwise hang the page's load event (dropdown stuck on "Loading models...").
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "vendor"))
    if os.path.exists(vendor_dir):
        app.router.add_static("/vendor", vendor_dir, name="vendor")
        logger.info(f"Serving vendored JS from: {vendor_dir}")
    else:
        logger.warning(f"⚠️  Vendor directory not found: {vendor_dir}")

    if not test_mode:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)

    return app


def get_app_config_dir():
    """Get the application config directory following OS conventions"""
    import os
    from pathlib import Path

    # Follow XDG Base Directory spec on Linux, use OS-appropriate paths elsewhere
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "live-vlm-webui"
        else:
            # Linux/Unix (including Jetson)
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "live-vlm-webui"
            )
    else:
        # Windows
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "live-vlm-webui"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """Generate a self-signed SSL certificate if it doesn't exist"""
    import subprocess
    import os

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    import ssl
    from . import __version__

    parser = argparse.ArgumentParser(
        description="ConanAI SAM WebUI — NanoOWL detection + NanoSAM segmentation, live over WebRTC",
        epilog="Example:\n"
        "  python -m webui.server --port 7860 --prompt '[an owl, a glove, a frog]'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--auto-port", action="store_true",
                        help="Automatically find an available port if default is taken")
    parser.add_argument("--owl-engine", default="/data/owl_image_encoder_patch32.engine",
                        help="Path to the NanoOWL image-encoder TRT engine")
    parser.add_argument("--owl-model", default="google/owlvit-base-patch32",
                        help="HF model name matching --owl-engine (patch32/patch16/large-patch14)")
    parser.add_argument("--sam-encoder", default="/data/nanosam/data/resnet18_image_encoder.engine",
                        help="Path to the NanoSAM image-encoder TRT engine")
    parser.add_argument("--sam-decoder", default="/data/nanosam/data/mobile_sam_mask_decoder.engine",
                        help="Path to the NanoSAM mask-decoder TRT engine")
    parser.add_argument("--no-sam", action="store_true", help="Disable NanoSAM, run OWL boxes only")
    parser.add_argument("--mask-alpha", type=float, default=0.5, help="Mask overlay alpha (0..1)")
    parser.add_argument("--prompt", default="[a person]",
                        help="NanoOWL tree prompt, e.g. \"[an owl, a glove, a frog]\"")
    parser.add_argument("--owl-threshold", type=float, default=0.1,
                        help="NanoOWL confidence threshold (default 0.1; lower → more detections)")

    default_config_dir = get_app_config_dir()
    default_cert_path = str(default_config_dir / "cert.pem")
    default_key_path = str(default_config_dir / "key.pem")

    parser.add_argument("--process-every", type=int, default=6,
                        help="Run inference every Nth incoming frame (default: 6)")
    parser.add_argument("--ssl-cert", default=None,
                        help=f"Path to SSL certificate file (default: {default_cert_path}, auto-generated if missing)")
    parser.add_argument("--ssl-key", default=None,
                        help=f"Path to SSL private key file (default: {default_key_path}, auto-generated if missing)")
    parser.add_argument("--no-ssl", action="store_true",
                        help="Disable SSL (browser getUserMedia requires HTTPS)")

    args = parser.parse_args()

    # Env overrides (CONANAI_*; legacy LIVE_VLM_PROCESS_EVERY honored for parity with vlm2)
    if os.environ.get("CONANAI_SAM_OWL_ENGINE"):
        args.owl_engine = os.environ["CONANAI_SAM_OWL_ENGINE"].strip()
    if os.environ.get("CONANAI_SAM_PROMPT"):
        args.prompt = os.environ["CONANAI_SAM_PROMPT"].strip()
    if os.environ.get("CONANAI_SAM_OWL_THRESHOLD"):
        try:
            args.owl_threshold = float(os.environ["CONANAI_SAM_OWL_THRESHOLD"])
        except ValueError:
            pass
    for env_key in ("LIVE_VLM_PROCESS_EVERY", "CONANAI_SAM_PROCESS_EVERY"):
        if os.environ.get(env_key):
            try:
                args.process_every = int(os.environ[env_key])
            except ValueError:
                pass

    if args.ssl_cert is None:
        args.ssl_cert = str(get_app_config_dir() / "cert.pem")
    if args.ssl_key is None:
        args.ssl_key = str(get_app_config_dir() / "key.pem")

    # Validate engines exist
    if not os.path.exists(args.owl_engine):
        logger.error(f"OWL engine not found: {args.owl_engine}")
        logger.error("Run container_setup.sh to build it.")
        sys.exit(1)
    sam_enc = None if args.no_sam else args.sam_encoder
    sam_dec = None if args.no_sam else args.sam_decoder
    if sam_enc and not os.path.exists(sam_enc):
        logger.warning(f"SAM encoder missing — running boxes-only: {sam_enc}")
        sam_enc = sam_dec = None
    if sam_dec and not os.path.exists(sam_dec):
        logger.warning(f"SAM decoder missing — running boxes-only: {sam_dec}")
        sam_enc = sam_dec = None

    global vlm_service, default_vlm_config
    vlm_service = OwlSamService(
        owl_engine=args.owl_engine,
        sam_image_encoder_engine=sam_enc,
        sam_mask_decoder_engine=sam_dec,
        prompt=args.prompt,
        mask_alpha=args.mask_alpha,
        owl_threshold=args.owl_threshold,
        owl_model_name=args.owl_model,
    )
    default_vlm_config = {
        "owl_engine": args.owl_engine,
        "owl_model_name": args.owl_model,
        "sam_image_encoder_engine": sam_enc,
        "sam_mask_decoder_engine": sam_dec,
        "prompt": args.prompt,
        "mask_alpha": args.mask_alpha,
        "owl_threshold": args.owl_threshold,
        # Kept so template code that still expects these doesn't crash:
        "model": "nanoowl+nanosam",
        "api_base": "local",
        "api_key": "N/A",
    }
    sessions["default"] = {
        "vlm_service": vlm_service,
        "show_request_payload": False,
        "show_response_payload": False,
    }

    logger.info("Initialized OWL+SAM service:")
    logger.info(f"  OWL engine: {args.owl_engine}")
    logger.info(f"  OWL threshold: {args.owl_threshold}")
    logger.info(f"  SAM enc/dec: {sam_enc} / {sam_dec}")
    logger.info(f"  Prompt: {args.prompt}")

    # Update frame processing rate in VideoProcessorTrack if needed
    # (This is a bit hacky but works for this demo)
    VideoProcessorTrack.process_every_n_frames = args.process_every

    # Create web application using create_app
    app = asyncio.run(create_app(test_mode=False))

    # Setup SSL (auto-generate certificates if needed)
    ssl_context = None
    protocol = "http"
    if not args.no_ssl:
        # Try to auto-generate if certificates don't exist
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            success = generate_self_signed_cert(args.ssl_cert, args.ssl_key)
            if not success:
                # FAIL FAST - SSL is required for webcam access
                logger.error("")
                logger.error("❌ Cannot start server without SSL certificates")
                logger.error("❌ Webcam access requires HTTPS!")
                logger.error("")
                logger.error("🔧 To fix, install openssl:")
                logger.error("   Linux/Jetson: sudo apt install openssl")
                logger.error("   macOS: brew install openssl")
                logger.error("")
                logger.error("   Then restart the server")
                logger.error("")
                logger.error(
                    "⚠️  Or run with --no-ssl if you don't need camera access (not recommended)"
                )
                logger.error("")
                sys.exit(1)

        # Load certificates (they must exist at this point)
        if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
            protocol = "https"
            logger.info("SSL enabled - using HTTPS")
        else:
            # This should never happen, but just in case
            logger.error("❌ SSL certificates missing after generation - unexpected error")
            sys.exit(1)
    else:
        logger.warning("⚠️  SSL disabled with --no-ssl flag")
        logger.warning("⚠️  Webcam access will NOT work without HTTPS!")

    # Get network addresses
    import socket
    import subprocess

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Access the server at:")
    logger.info(f"  Local:   {protocol}://localhost:{args.port}")

    # Get network interfaces - try multiple methods for cross-platform support
    network_ips = []

    # Method 1: hostname -I (Linux)
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                # Filter out loopback and docker bridges (172.17.x.x)
                if not ip.startswith("127.") and not ip.startswith("172.17."):
                    network_ips.append(ip)
    except Exception:
        pass

    # Method 2: Socket method (cross-platform fallback)
    if not network_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1":
                network_ips.append(ip)
        except Exception:
            pass

    # Display all found network IPs
    for ip in network_ips:
        logger.info(f"  Network: {protocol}://{ip}:{args.port}")

    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


def stop():
    """Stop the running live-vlm-webui server"""
    import sys
    import time

    try:
        import psutil
    except ImportError:
        logger.error("psutil is required for the stop command")
        logger.error("Install it with: pip install live-vlm-webui[dev]")
        sys.exit(1)

    print("Stopping Live VLM WebUI server...")

    # Find and kill processes running live_vlm_webui.server
    found = False
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    # Don't kill the stop command itself
                    if "stop" not in cmdline_str:
                        found = True
                        print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                        proc.terminate()
                        killed.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("✓ No running server found")
        return

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for proc in killed:
        try:
            if proc.is_running():
                print(f"  Force killing process {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Final verification
    time.sleep(1)
    still_running = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    if "stop" not in cmdline_str:
                        still_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if still_running:
        print("❌ Failed to stop server")
        sys.exit(1)
    else:
        print("✓ Server stopped successfully")


if __name__ == "__main__":
    main()
