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
Video Track Processor
Handles video frames, adds text overlays, and manages OWL+SAM processing
"""

import asyncio
import cv2
import numpy as np
from PIL import Image
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from typing import Optional
import logging
import time
import av

from .owl_sam_service import OwlSamService as VLMService  # noqa: N814  (drop-in for VLMService)

# Enable swscaler warnings to track hardware acceleration status
# TODO: Implement hardware-accelerated color space conversion on Jetson using NVMM/VPI
av.logging.set_level(av.logging.WARNING)

logger = logging.getLogger(__name__)


class VideoProcessorTrack(VideoStreamTrack):
    """
    Video track that receives frames, sends them to OWL+SAM for analysis,
    and overlays responses on the video before sending back
    """

    # Class variable for frame processing interval (can be updated dynamically)
    process_every_n_frames = 30
    # Max allowed latency before dropping frames (in seconds, 0 = disabled)
    max_frame_latency = 0.0

    # --- Analysis cadence ---------------------------------------------------
    # Controls WHEN frames are dispatched to OWL+SAM. Default preserves the
    # original frame-based behavior. Settable at runtime via the /offer body.
    #
    #   "frames"        fire every process_every_n_frames frames (default).
    #   "seconds"       fire every analysis_interval_seconds wall-clock seconds.
    #   "scene_change"  fire only when consecutive frames differ by more than
    #                   scene_change_threshold, with at least
    #                   scene_change_min_interval seconds between fires
    #                   (debounce / Ollama DoS guard).
    analysis_mode = "frames"
    analysis_interval_seconds = 5.0
    scene_change_threshold = 15.0  # 0-255 mean absolute diff on 64x64 grayscale
    scene_change_min_interval = 2.0  # seconds; lower bound between dispatches

    def __init__(self, track: VideoStreamTrack, vlm_service: VLMService, text_callback=None):
        super().__init__()
        self.track = track
        self.vlm_service = vlm_service
        self.text_callback = text_callback  # Callback to send text updates
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.first_frame_pts = None  # Track first frame PTS to calculate relative time
        self.first_frame_time = None  # Wall clock time of first frame
        self.frame_time_base = None  # Time base for PTS conversion (e.g., 1/90000)
        # Analysis cadence state
        self._last_analysis_time: float = 0.0
        self._prev_thumbnail: Optional[np.ndarray] = None
        self._last_scene_change_score: float = 0.0

    async def recv(self):
        """
        Receive frame from input track, process it, and return with text overlay
        """
        try:
            # Get frame from incoming track
            frame = await self.track.recv()

            # Initialize timing on first frame
            if self.first_frame_pts is None and frame.pts is not None:
                self.first_frame_pts = frame.pts
                self.first_frame_time = time.time()
                # Store time_base for PTS conversion (e.g., 1/90000 for 90kHz clock)
                self.frame_time_base = float(frame.time_base)
                logger.info(
                    f"Latency tracking initialized: PTS={frame.pts}, time_base={frame.time_base} ({self.frame_time_base}s per tick)"
                )

            # Calculate actual frame age (latency) using PTS and time_base
            # Note: Some streams (like RTSP) may not have PTS set, so skip latency checks
            frame_latency = 0.0
            if frame.pts is not None and self.first_frame_pts is not None:
                # PTS is in time_base units, convert to seconds: pts * time_base
                frame_time_offset = (frame.pts - self.first_frame_pts) * self.frame_time_base
                expected_wall_time = self.first_frame_time + frame_time_offset
                current_time = time.time()
                frame_latency = current_time - expected_wall_time

            # Check for accumulated latency and drop old frames if needed (only if max_latency > 0)
            max_latency = self.__class__.max_frame_latency
            if max_latency > 0 and frame_latency > max_latency and frame.pts is not None:
                logger.warning(
                    f"Frame is {frame_latency:.2f}s behind, dropping frames (threshold: {max_latency}s)"
                )

                # Drop frames until we get a fresh one
                dropped_count = 0
                while frame_latency > max_latency:
                    self.dropped_frames += 1
                    dropped_count += 1

                    # Get next frame
                    frame = await self.track.recv()

                    # Recalculate latency for new frame (using time_base for correct conversion)
                    if frame.pts is not None and self.first_frame_pts is not None:
                        frame_time_offset = (
                            frame.pts - self.first_frame_pts
                        ) * self.frame_time_base
                        expected_wall_time = self.first_frame_time + frame_time_offset
                        frame_latency = time.time() - expected_wall_time
                    else:
                        # If PTS becomes unavailable, stop dropping frames
                        break

                    # Prevent infinite loop
                    if dropped_count > 100:
                        logger.error(
                            f"Dropped {dropped_count} frames, but still behind. Resetting timing."
                        )
                        if frame.pts is not None:
                            self.first_frame_pts = frame.pts
                            self.first_frame_time = time.time()
                            self.frame_time_base = float(frame.time_base)
                        break

                if dropped_count > 0:
                    logger.info(
                        f"Dropped {dropped_count} frames, now at {frame_latency:.2f}s latency"
                    )

            # Increment frame counter
            self.frame_count += 1

            # ---------------------------------------------------------------
            # Decide whether to dispatch this frame to OWL+SAM based on the
            # currently-selected analysis_mode. BGR conversion is done lazily
            # — only when needed for scene-change detection OR dispatch.
            # ---------------------------------------------------------------
            mode = self.__class__.analysis_mode
            interval_frames = self.__class__.process_every_n_frames
            interval_seconds = self.__class__.analysis_interval_seconds
            now = time.time()

            should_dispatch = False
            img = None  # BGR ndarray, computed once if needed

            if mode == "frames":
                if self.frame_count % interval_frames == 0:
                    should_dispatch = True

            elif mode == "seconds":
                # Wall-clock interval. First frame always fires so the user
                # sees immediate feedback on a new source.
                if (self._last_analysis_time == 0.0
                        or now - self._last_analysis_time >= interval_seconds):
                    should_dispatch = True

            elif mode == "scene_change":
                # Cheap thumbnail-diff (~1ms per frame). Reliable for
                # slideshow content where transitions are stark.
                img = frame.to_ndarray(format="bgr24")
                thumb = cv2.resize(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 64),
                    interpolation=cv2.INTER_AREA,
                )
                if self._prev_thumbnail is None:
                    should_dispatch = True
                    self._last_scene_change_score = 0.0
                else:
                    diff = float(
                        np.mean(np.abs(thumb.astype(np.int16)
                                       - self._prev_thumbnail.astype(np.int16)))
                    )
                    self._last_scene_change_score = diff
                    if diff >= self.__class__.scene_change_threshold:
                        # Debounce: don't fire more than once per
                        # scene_change_min_interval seconds.
                        if (now - self._last_analysis_time
                                >= self.__class__.scene_change_min_interval):
                            should_dispatch = True
                            logger.info(
                                f"Scene change detected (score={diff:.1f}, "
                                f"threshold={self.__class__.scene_change_threshold}); "
                                f"dispatching frame {self.frame_count}"
                            )
                self._prev_thumbnail = thumb

            else:
                logger.warning(
                    f"Unknown analysis_mode {mode!r}, falling back to 'frames'"
                )
                if self.frame_count % interval_frames == 0:
                    should_dispatch = True

            # Convert to ndarray if we haven't already and we need it.
            need_conversion = should_dispatch or (self.frame_count == 1)
            if need_conversion and img is None:
                t1 = time.time()
                img = frame.to_ndarray(format="bgr24")
                t2 = time.time()
                if self.frame_count % 100 == 0:
                    logger.debug(
                        f"Frame conversion: to_ndarray={1000*(t2-t1):.1f}ms"
                    )

            if img is not None:
                self.last_frame = img.copy() if should_dispatch else img
                if self.frame_count == 1:
                    logger.info(f"First frame received: {img.shape}")

            if should_dispatch:
                # Backpressure: skip if the previous OWL+SAM call is still
                # running. Without this, slow inference piles up tasks until
                # something OOMs or hangs — which is the "inference stops
                # responding" symptom emplus exhibits.
                # vlm_service.process_frame() also guards via a lock; this
                # check just lets us log the skip and not create the task.
                if getattr(self.vlm_service, "is_processing", False):
                    logger.info(
                        f"Frame {self.frame_count}: skip dispatch — "
                        f"OWL+SAM still processing previous frame (mode={mode})"
                    )
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    asyncio.create_task(self.vlm_service.process_frame(pil_img))
                    self._last_analysis_time = now
                    logger.info(
                        f"Frame {self.frame_count}: dispatched to OWL+SAM (mode={mode})"
                    )

            # Get current response (may be old if OWL+SAM is still processing)
            response, is_processing = self.vlm_service.get_current_response()

            # Get metrics
            metrics = self.vlm_service.get_metrics()

            # Send text update via callback (for WebSocket)
            if self.text_callback:
                self.text_callback(response, metrics)

            # MODE_TRACK_SAM: draw motion-shifted masks onto the LIVE frame every frame, so the
            # video plays at full fps with masks that follow the object between inferences —
            # instead of freezing on the last baked annotation.
            if (hasattr(self.vlm_service, "wants_live_overlay")
                    and self.vlm_service.wants_live_overlay()):
                try:
                    live_bgr = frame.to_ndarray(format="bgr24")
                    out = self.vlm_service.overlay_live(live_bgr)
                    new_frame = av.VideoFrame.from_ndarray(out, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    return new_frame
                except Exception as e:
                    logger.warning(f"Live overlay failed: {e}")
                    return frame

            # If the inference service has produced an annotated frame (NanoOWL boxes
            # + NanoSAM masks baked in), swap it in so the browser sees the overlay.
            # While SAM is still warming up (first inference pending), fall through to
            # the raw frame so the video stays live instead of going black.
            annotated = self.vlm_service.get_last_annotation() if hasattr(
                self.vlm_service, "get_last_annotation"
            ) else None
            if annotated is not None:
                try:
                    new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    return new_frame
                except Exception as e:
                    logger.warning(f"Annotated frame swap failed: {e}")
            return frame

        except MediaStreamError:
            # Track ended (user stopped, tab closed, etc.) — normal, not an error
            logger.debug("Video track ended")
            raise
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            raise

    def _add_text_overlay(self, img: np.ndarray, text: str, status: str = "") -> np.ndarray:
        """
        Add text overlay to image

        Args:
            img: Input image (BGR format)
            text: Text to overlay (OWL+SAM response)
            status: Optional status text

        Returns:
            Image with text overlay
        """
        img_copy = img.copy()
        height, width = img_copy.shape[:2]

        # Prepare text
        full_text = f"{text} {status}" if status else text

        # Text wrapping - split long captions
        max_chars_per_line = 60
        words = full_text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background
        padding = 10
        line_height = 30

        # Calculate total height needed
        total_text_height = len(lines) * line_height + 2 * padding

        # Create semi-transparent overlay at bottom
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, height - total_text_height), (width, height), bg_color, -1)

        # Blend overlay with original image
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

        # Add text lines
        y_position = height - total_text_height + padding + line_height
        for line in lines:
            cv2.putText(
                img_copy,
                line,
                (padding, y_position),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            y_position += line_height

        return img_copy
