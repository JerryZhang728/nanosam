"""
Local File Video Track

Provides a VideoStreamTrack that reads from local files instead of network sources.
Supports three modes (auto-detected by path):

  * Single image file (.png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp)
        Emits the same frame repeatedly at a configurable FPS.

  * Folder of images (slideshow)
        Cycles through sorted image files, holding each for a configurable
        number of seconds before advancing. Loops back to the first image.

  * Video file (.mp4, .avi, .mkv, .mov, .wmv, .flv, .webm)
        Decodes via PyAV, paced to the file's native FPS. On EOF, optionally
        seeks back to frame 0 and loops.

The track exposes the same surface as RTSPVideoTrack: ``recv()``, ``stop()``,
``is_connected``, ``get_stats()``.

SPDX-FileCopyrightText: Copyright (c) 2026 Live VLM WebUI contributors.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio
import fractions
import logging
import time
from pathlib import Path
from typing import List, Optional

import av
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from PIL import Image

# Mute PyAV's verbose decoder warnings; only show fatal errors.
av.logging.set_level(av.logging.FATAL)

logger = logging.getLogger(__name__)

# Video time base: 90 kHz is the WebRTC standard for video RTP.
VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class LocalFileVideoTrack(VideoStreamTrack):
    """
    Video track that sources frames from a local image, image folder, or video file.

    Args:
        path: Absolute path to an image file, image directory, or video file.
        image_fps: Frames per second when the source is a still image. Default 1.0.
            Cheap and predictable; analysis cadence is controlled separately by
            the processor's analysis_mode / analysis_interval settings.
        slideshow_seconds_per_image: How long each image is held in slideshow
            mode before advancing to the next. Default 5.0.
        slideshow_fps: Output frame rate while showing each slideshow image.
            Independent of how long each image is held. Default 1.0.
        loop: Whether to restart at frame 0 when a video file reaches EOF.
            Slideshow mode always loops. Has no effect on single-image mode
            (which emits forever by definition). Default True.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not supported, or if a folder is
            given but contains no images.
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

    def __init__(
        self,
        path: str,
        image_fps: float = 1.0,
        slideshow_seconds_per_image: float = 5.0,
        slideshow_fps: float = 1.0,
        loop: bool = True,
    ):
        super().__init__()
        self.path = Path(path)
        self.image_fps = max(0.1, float(image_fps))
        self.slideshow_seconds_per_image = max(0.5, float(slideshow_seconds_per_image))
        self.slideshow_fps = max(0.1, float(slideshow_fps))
        self.loop = bool(loop)

        self._stopped = False
        self._frame_count = 0
        self._start_wall_time: Optional[float] = None

        # Mode-specific state, set in _init_mode()
        self.mode: str = ""
        self._still_frame: Optional[VideoFrame] = None
        self._image_files: List[Path] = []
        self._current_image_idx: int = 0
        self._current_image_started_at: Optional[float] = None
        self._container: Optional[av.container.InputContainer] = None
        self._stream: Optional[av.video.VideoStream] = None
        self._native_fps: float = 25.0
        self._loops_completed: int = 0

        self._init_mode()

    # ------------------------------------------------------------------ init

    def _init_mode(self) -> None:
        """Inspect the path, decide the mode, and open the source."""
        if not self.path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.path}")

        if self.path.is_dir():
            self._init_slideshow()
        else:
            suffix = self.path.suffix.lower()
            if suffix in self.IMAGE_EXTS:
                self._init_image()
            elif suffix in self.VIDEO_EXTS:
                self._init_video()
            else:
                raise ValueError(
                    f"Unsupported file type {suffix!r}. "
                    f"Supported images: {sorted(self.IMAGE_EXTS)}, "
                    f"videos: {sorted(self.VIDEO_EXTS)}."
                )

    def _init_image(self) -> None:
        self.mode = "image"
        self._still_frame = self._load_image_as_frame(self.path)
        logger.info(
            f"LocalFileVideoTrack: image mode, {self.path.name} "
            f"({self._still_frame.width}x{self._still_frame.height}) @ {self.image_fps} fps"
        )

    def _init_slideshow(self) -> None:
        self.mode = "slideshow"
        self._image_files = sorted(
            p
            for p in self.path.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self._image_files:
            raise ValueError(
                f"Folder contains no supported images: {self.path}. "
                f"Supported extensions: {sorted(self.IMAGE_EXTS)}"
            )
        self._current_image_idx = 0
        self._still_frame = self._load_image_as_frame(self._image_files[0])
        logger.info(
            f"LocalFileVideoTrack: slideshow mode, {len(self._image_files)} images "
            f"in {self.path}, holding each for {self.slideshow_seconds_per_image}s "
            f"@ {self.slideshow_fps} fps output"
        )

    def _init_video(self) -> None:
        self.mode = "video"
        self._open_container()
        codec = self._stream.codec_context.name if self._stream else "?"
        w = self._stream.width if self._stream else 0
        h = self._stream.height if self._stream else 0
        logger.info(
            f"LocalFileVideoTrack: video mode, {self.path.name} "
            f"({codec} {w}x{h} @{self._native_fps:.2f}fps, loop={self.loop})"
        )

    def _open_container(self) -> None:
        """Open or reopen the PyAV container for video mode."""
        if self._container is not None:
            try:
                self._container.close()
            except Exception as e:
                logger.debug(f"Error closing existing container: {e}")
        self._container = av.open(str(self.path))
        if not self._container.streams.video:
            raise ValueError(f"No video stream found in: {self.path}")
        self._stream = self._container.streams.video[0]
        rate = self._stream.average_rate or self._stream.guessed_rate
        self._native_fps = float(rate) if rate else 25.0
        if self._native_fps <= 0 or self._native_fps > 240:
            logger.warning(
                f"Suspicious native FPS {self._native_fps}, defaulting to 25.0"
            )
            self._native_fps = 25.0

    @staticmethod
    def _load_image_as_frame(image_path: Path) -> VideoFrame:
        """Load an image file into an aiortc-compatible VideoFrame (rgb24)."""
        with Image.open(image_path) as pil_img:
            img = pil_img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
        return VideoFrame.from_ndarray(arr, format="rgb24")

    # ------------------------------------------------------------------ recv

    async def recv(self) -> VideoFrame:
        """
        Return the next frame, paced to match the requested output rate.

        WebRTC consumers (aiortc) call this in a tight loop; we throttle by
        sleeping so that frames go out at the configured FPS rather than as
        fast as the file can be decoded.
        """
        if self._stopped:
            raise StopAsyncIteration

        if self._start_wall_time is None:
            self._start_wall_time = time.time()

        try:
            if self.mode == "image":
                frame = await self._recv_image()
            elif self.mode == "slideshow":
                frame = await self._recv_slideshow()
            elif self.mode == "video":
                frame = await self._recv_video()
            else:
                raise RuntimeError(f"Unknown mode: {self.mode!r}")
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error in LocalFileVideoTrack.recv: {e}", exc_info=True)
            raise

        # Stamp WebRTC pts/time_base. Use frame_count at the output rate as the
        # monotonic clock. The 90 kHz WebRTC video clock is standard.
        output_fps = self._current_output_fps()
        frame.pts = int(self._frame_count * (VIDEO_CLOCK_RATE / output_fps))
        frame.time_base = VIDEO_TIME_BASE

        self._frame_count += 1
        return frame

    def _current_output_fps(self) -> float:
        if self.mode == "image":
            return self.image_fps
        if self.mode == "slideshow":
            return self.slideshow_fps
        return self._native_fps

    async def _pace_to_rate(self, fps: float) -> None:
        """Sleep so frames are emitted at ``fps`` based on wall clock."""
        if fps <= 0:
            return
        target = self._start_wall_time + (self._frame_count / fps)
        sleep_for = target - time.time()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    async def _recv_image(self) -> VideoFrame:
        await self._pace_to_rate(self.image_fps)
        # Return a NEW VideoFrame wrapping the same ndarray so we don't mutate
        # the cached one's pts/time_base (PyAV frames are mutable).
        return VideoFrame.from_ndarray(
            self._still_frame.to_ndarray(format="rgb24"), format="rgb24"
        )

    async def _recv_slideshow(self) -> VideoFrame:
        now = time.time()
        if self._current_image_started_at is None:
            self._current_image_started_at = now

        # Advance to the next image when its hold time has elapsed.
        if now - self._current_image_started_at >= self.slideshow_seconds_per_image:
            self._current_image_idx = (self._current_image_idx + 1) % len(self._image_files)
            self._current_image_started_at = now
            self._still_frame = self._load_image_as_frame(
                self._image_files[self._current_image_idx]
            )
            logger.debug(
                f"Slideshow advanced to image {self._current_image_idx + 1}/"
                f"{len(self._image_files)}: "
                f"{self._image_files[self._current_image_idx].name}"
            )

        await self._pace_to_rate(self.slideshow_fps)
        return VideoFrame.from_ndarray(
            self._still_frame.to_ndarray(format="rgb24"), format="rgb24"
        )

    async def _recv_video(self) -> VideoFrame:
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self._read_video_frame)

        if frame is None:
            if self.loop:
                self._loops_completed += 1
                logger.debug(f"Video EOF, looping (loop #{self._loops_completed})")
                await loop.run_in_executor(None, self._seek_to_start)
                frame = await loop.run_in_executor(None, self._read_video_frame)
                if frame is None:
                    # If we can't read even after seeking, the file is broken
                    # or empty; give up.
                    raise StopAsyncIteration
            else:
                logger.info(f"Video reached EOF, stopping (loop=False)")
                raise StopAsyncIteration

        await self._pace_to_rate(self._native_fps)
        return frame

    def _read_video_frame(self) -> Optional[VideoFrame]:
        """Blocking video frame read. Run in executor."""
        if not self._container or not self._stream:
            return None
        try:
            for packet in self._container.demux(self._stream):
                for f in packet.decode():
                    if isinstance(f, VideoFrame):
                        return f
            return None  # End of file
        except av.error.EOFError:
            return None
        except Exception as e:
            logger.error(f"Error decoding video frame: {e}")
            return None

    def _seek_to_start(self) -> None:
        """Seek to the beginning of the video (for looping)."""
        try:
            self._container.seek(0, stream=self._stream)
        except Exception as e:
            logger.warning(f"Seek failed ({e}); reopening container")
            self._open_container()

    # ----------------------------------------------------------- lifecycle

    def stop(self) -> None:
        """Release resources. Idempotent."""
        if self._stopped:
            super().stop()
            return
        self._stopped = True

        if self._container is not None:
            try:
                self._container.close()
                logger.info(
                    f"LocalFileVideoTrack stopped: {self._frame_count} frames emitted "
                    f"(mode={self.mode}, loops={self._loops_completed})"
                )
            except Exception as e:
                logger.warning(f"Error closing container: {e}")
            finally:
                self._container = None
                self._stream = None
        else:
            logger.info(
                f"LocalFileVideoTrack stopped: {self._frame_count} frames emitted "
                f"(mode={self.mode})"
            )

        super().stop()

    @property
    def is_connected(self) -> bool:
        return not self._stopped

    def get_stats(self) -> dict:
        stats = {
            "mode": self.mode,
            "path": str(self.path),
            "frames_emitted": self._frame_count,
            "stopped": self._stopped,
            "loop": self.loop,
        }
        if self.mode == "image":
            stats["image_fps"] = self.image_fps
            if self._still_frame is not None:
                stats["width"] = self._still_frame.width
                stats["height"] = self._still_frame.height
        elif self.mode == "slideshow":
            stats["slideshow_count"] = len(self._image_files)
            stats["current_index"] = self._current_image_idx
            stats["current_image"] = (
                self._image_files[self._current_image_idx].name
                if self._image_files
                else None
            )
            stats["seconds_per_image"] = self.slideshow_seconds_per_image
            stats["slideshow_fps"] = self.slideshow_fps
        elif self.mode == "video":
            stats["native_fps"] = self._native_fps
            stats["loops_completed"] = self._loops_completed
            if self._stream is not None:
                stats["codec"] = self._stream.codec_context.name
                stats["width"] = self._stream.width
                stats["height"] = self._stream.height
        return stats
