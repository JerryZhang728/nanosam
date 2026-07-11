# SPDX-FileCopyrightText: Copyright (c) 2026 ConanAI / nanosam
# SPDX-FileCopyrightText: Portions Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# This module replaces live-vlm-webui's vlm_service.py with an in-process
# NanoOWL (text-prompted detection) + NanoSAM (box-prompted segmentation) service.
# It mirrors VLMService's public surface (process_frame / get_current_response /
# get_metrics / update_prompt / update_api_settings) so the rest of the live-vlm-webui
# server (server.py, video_processor.py) keeps working unchanged.
#
# Two things differ from VLMService:
#   1. There is no external API — inference is in-process via the NanoOWL + NanoSAM
#      TRT engines that container_setup.sh builds under /data.
#   2. We expose get_last_annotation() returning a BGR ndarray with boxes + masks
#      already drawn. VideoProcessorTrack uses it to replace the live frame so the
#      browser sees the annotated video (instead of overlaying text in HTML, which
#      is what VLMService does).

import asyncio
import logging
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import PIL.Image
import torch

logger = logging.getLogger(__name__)


_PALETTE = [
    (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 255, 255),
    (255, 56, 255), (255, 255, 56), (255, 128, 0), (128, 0, 255),
]


def _color(i: int):
    return _PALETTE[i % len(_PALETTE)]


# --- Memory management / leak diagnostics (tunable via env for A/B testing) ---
# SAM_EMPTY_CACHE_EVERY=N : call torch.cuda.empty_cache() every N inferences (0=off).
#   Default 1 (every inference). If the VRAM climb FLATTENS with this on, the leak
#   was CUDA allocator *fragmentation* — empty_cache is the fix. If it STILL climbs,
#   it's a true leak (held tensors) or TRT-side raw cudaMalloc that torch can't free.
# SAM_MEM_LOG_EVERY=N : log torch allocated vs reserved every N inferences (0=off).
#   allocated grows  -> real tensor leak (references retained).
#   allocated flat, reserved grows -> fragmentation (empty_cache helps).
#   both flat but tegrastats still climbs -> leak OUTSIDE torch (TensorRT / CPU).
# Default OFF: on-device diagnosis showed torch stays flat at ~1GB (alloc≈reserved,
# no leak, no fragmentation) — the real leak was the aiortc relay's unbounded frame
# buffer (fixed in server.py with buffered=False). empty_cache per-frame just adds a
# device sync for no benefit here. Left as a knob in case a future model fragments.
_EMPTY_CACHE_EVERY = int(os.environ.get("SAM_EMPTY_CACHE_EVERY", "0"))
_MEM_LOG_EVERY = int(os.environ.get("SAM_MEM_LOG_EVERY", "20"))

# Max objects segmented per frame in the tracked-mask mode (each = a SAM decode + a stored,
# per-frame-animated full-frame mask). A big shoal (40+) OOMs/aborts the process; cap it. Tunable.
_MAX_SEG = int(os.environ.get("SAM_MAX_OBJECTS", "20"))


# Modes — selected via the model dropdown in the UI.
MODE_OWL_SAM = "nanoowl+nanosam"      # OWL detect → SAM segment per box
MODE_OWL_ONLY = "nanoowl"             # OWL detect → boxes only (skip SAM)
MODE_OWL_TRACK = "nanoowl+bytetrack"  # OWL detect → ByteTrack → boxes with stable IDs
MODE_TRACK_SAM = "nanoowl+bytetrack+nanosam"  # OWL → ByteTrack → SAM: mask-only, per-track color,
                                              # motion-shifted between inferences (no box/ID drawn)
MODE_SAM_ONLY = "nanosam"             # SAM full-frame box prompt (skip OWL) — diagnostic, not in UI
VALID_MODES = {MODE_OWL_SAM, MODE_OWL_ONLY, MODE_OWL_TRACK, MODE_TRACK_SAM, MODE_SAM_ONLY}


class OwlSamService:
    """NanoOWL + NanoSAM inference service shaped like VLMService."""

    def __init__(
        self,
        owl_engine: str,
        sam_image_encoder_engine: Optional[str] = None,
        sam_mask_decoder_engine: Optional[str] = None,
        prompt: str = "[a person]",
        mask_alpha: float = 0.5,
        owl_threshold: float = 0.1,
        owl_model_name: str = "google/owlvit-base-patch32",
        # The next args exist only so server.py's VLMService(model=..., api_base=...)
        # call sites can reach us without changes. `model` doubles as the mode selector.
        # Default to OWL+SAM (masks) — the headline demo; users switch modes in the UI.
        model: str = MODE_OWL_SAM,
        api_base: str = "local",
        api_key: str = "N/A",
        max_tokens: int = 0,
    ):
        self.owl_engine = owl_engine
        self.sam_image_encoder_engine = sam_image_encoder_engine
        self.sam_mask_decoder_engine = sam_mask_decoder_engine
        self.prompt = prompt
        self.mask_alpha = mask_alpha
        self.owl_threshold = owl_threshold
        self.owl_model_name = owl_model_name

        # VLMService-compatible attributes (server.py reads these)
        self._mode = model if model in VALID_MODES else MODE_OWL_SAM
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens

        # State
        self.current_response = "Initializing..."
        self.is_processing = False
        self._processing_lock = asyncio.Lock()
        self._last_request_payload = None
        self._last_response_payload = None
        self._last_annotation: Optional[np.ndarray] = None

        # Metrics
        self.last_inference_time = 0.0
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self._infer_calls = 0  # local counter for cache/mem-log cadence

        # Lazily-initialized predictors (loading TRT engines takes seconds)
        self._owl = None
        self._sam = None
        self._tracker = None            # supervision.ByteTrack, created on first use in track mode
        self._tracker_unavailable = False
        # For MODE_TRACK_SAM: per-track {mask_bool, center, vel_px_per_s, t, color}. Updated on
        # inference frames; read (and motion-shifted) every display frame by overlay_live().
        self._track_masks = {}
        # For MODE_OWL_TRACK: per-track {box, center, vel, t, color} — lets the display coast
        # the boxes onto the live frame between inferences (smooth video, no freeze).
        self._track_boxes = {}
        self._tree = None
        self._clip_enc = None
        self._owl_enc = None
        self._encoded_prompt: Optional[str] = None

    # ---- Lazy predictor initialization ----
    def _ensure_predictors(self):
        if self._owl is None:
            logger.info(
                f"Loading NanoOWL TreePredictor (model={self.owl_model_name}, "
                f"engine={self.owl_engine})..."
            )
            from nanoowl.tree_predictor import TreePredictor
            from nanoowl.owl_predictor import OwlPredictor
            self._owl = TreePredictor(
                owl_predictor=OwlPredictor(
                    model_name=self.owl_model_name,
                    image_encoder_engine=self.owl_engine,
                )
            )
            logger.info("NanoOWL ready.")

        if self._sam is None and self.sam_image_encoder_engine and self.sam_mask_decoder_engine:
            try:
                logger.info("Loading NanoSAM Predictor (one-time)...")
                from nanosam.utils.predictor import Predictor as SamPredictor
                self._sam = SamPredictor(
                    self.sam_image_encoder_engine,
                    self.sam_mask_decoder_engine,
                )
                logger.info("NanoSAM ready.")
            except Exception as e:
                logger.warning(f"NanoSAM unavailable — running OWL only: {e}")
                self._sam = None

    def _ensure_prompt_encoded(self):
        if self._encoded_prompt == self.prompt and self._tree is not None:
            return
        from nanoowl.tree import Tree
        self._tree = Tree.from_prompt(self.prompt)
        self._clip_enc = self._owl.encode_clip_text(self._tree)
        self._owl_enc = self._owl.encode_owl_text(self._tree)
        self._encoded_prompt = self.prompt
        logger.info(f"Encoded prompt: {self.prompt!r}")

    # ---- Mode selector (exposed as `.model` so the UI's model-dropdown plumbing works) ----
    @property
    def model(self):
        return self._mode

    @model.setter
    def model(self, new_value):
        if not new_value:
            return
        if new_value in VALID_MODES:
            old = self._mode
            self._mode = new_value
            if old != new_value:
                logger.info(f"Mode switched: {old} → {new_value}")
        else:
            logger.warning(f"Ignoring unknown mode/model: {new_value!r} (valid: {sorted(VALID_MODES)})")

    # ---- Inference ----
    def _run_inference(self, pil_image: PIL.Image.Image) -> Tuple[np.ndarray, str]:
        """Branch on self._mode. Returns (annotated_bgr, summary_text).

        The whole body runs under torch.inference_mode(): neither NanoOWL nor
        NanoSAM wraps its own forward passes, so with autograd left on, every
        frame's activation tensors are retained for a backward pass that never
        comes. On the Jetson's unified GPU/CPU memory that leaks ~8→15GB over a
        minute until it OOMs and throttles. inference_mode() disables autograd
        for both models at once, so nothing is retained frame-to-frame.
        """
        self._ensure_predictors()
        self._infer_calls += 1
        with torch.inference_mode():
            bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]

            if self._mode == MODE_SAM_ONLY:
                result = self._run_sam_only(pil_image, bgr, h, w)
            else:
                result = self._run_owl_path(pil_image, bgr, h, w)

        n = self._infer_calls
        if _EMPTY_CACHE_EVERY and n % _EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()
        if _MEM_LOG_EVERY and n % _MEM_LOG_EVERY == 0:
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            logger.info(f"[mem] inf#{n} torch_alloc={a:.2f}GB torch_reserved={r:.2f}GB")
        return result

    def _run_owl_path(self, pil_image, bgr, h, w):
        """OWL detect → (optionally) SAM segment per box → draw boxes."""
        from nanoowl.tree_drawing import draw_tree_output

        self._ensure_prompt_encoded()
        # NanoOWL's TreePredictor.predict accepts `threshold`; lower it to surface
        # more (lower-confidence) detections. Default 0.1 mirrors upstream.
        try:
            output = self._owl.predict(
                pil_image,
                tree=self._tree,
                clip_text_encodings=self._clip_enc,
                owl_text_encodings=self._owl_enc,
                threshold=self.owl_threshold,
            )
        except TypeError:
            # Older NanoOWL signatures may not accept threshold
            output = self._owl.predict(
                pil_image,
                tree=self._tree,
                clip_text_encodings=self._clip_enc,
                owl_text_encodings=self._owl_enc,
            )

        # Strip the implicit root "image" detection (covers the whole frame, score 1.0).
        output.detections = self._strip_root_detections(output.detections)

        # ByteTrack mode: assign stable IDs across frames, draw our own labelled boxes.
        if self._mode == MODE_OWL_TRACK:
            self._ensure_tracker()
            if self._tracker is not None:
                return self._track_and_draw(bgr, output)
            # supervision missing -> fall through to plain boxes below

        # Track + SAM mode: tracker IDs → SAM mask per tracked box, stored for motion-shifted
        # live overlay. Render mask-only (no box/ID). Needs both tracker and SAM.
        if self._mode == MODE_TRACK_SAM:
            self._ensure_tracker()
            if self._tracker is not None and self._sam is not None:
                return self._run_track_sam(pil_image, bgr, output, h, w)
            # missing tracker or SAM -> fall through (boxes / masks as available)

        if (
            self._mode == MODE_OWL_SAM
            and self._sam is not None
            and len(output.detections) > 0
        ):
            # Masks only — clean overlay, no boxes/labels drawn.
            bgr = self._overlay_sam_masks(pil_image, bgr, output, h, w)
            return bgr, self._owl_summary(output)

        # OWL-only (or SAM unavailable): draw boxes + labels.
        bgr = draw_tree_output(bgr, output, self._tree)
        return bgr, self._owl_summary(output)

    def _ensure_tracker(self):
        """Lazily create a supervision ByteTrack (param names vary across sv versions)."""
        if self._tracker is not None or self._tracker_unavailable:
            return
        try:
            import supervision as sv
            try:
                self._tracker = sv.ByteTrack(
                    track_activation_threshold=self.owl_threshold,
                    lost_track_buffer=30, frame_rate=30, minimum_consecutive_frames=1,
                )
            except TypeError:  # older supervision API
                self._tracker = sv.ByteTrack(track_thresh=self.owl_threshold,
                                             track_buffer=30, frame_rate=30)
            logger.info("ByteTrack ready.")
        except Exception as e:
            logger.warning(f"ByteTrack unavailable (supervision not installed?) — "
                           f"track mode falls back to plain boxes: {e}")
            self._tracker_unavailable = True

    def _track_and_draw(self, bgr, output):
        """Run OWL detections through ByteTrack; draw boxes labelled with stable IDs.
        ByteTrack bridges frames where the detector drops out and suppresses 1-frame flicker
        (its low-confidence second-association pass), so IDs stay stable on moving objects."""
        import supervision as sv
        dets = output.detections
        h, w = bgr.shape[:2]
        if dets:
            boxes = np.array([d.box for d in dets], dtype=float)
            scores = np.array([max(d.scores) if d.scores else 0.5 for d in dets], dtype=float)
            class_id = np.array([d.labels[0] if d.labels else 0 for d in dets], dtype=int)
            sv_det = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_id)
        else:
            sv_det = sv.Detections.empty()
        tracked = self._tracker.update_with_detections(sv_det)
        tboxes = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tboxes.append((int(tracked.tracker_id[i]), tracked.xyxy[i]))

        # Draw a box for EVERY OWL detection (accuracy == plain OWL — the tracker does NOT gate).
        # Borrow a stable id + coasting velocity from ByteTrack where it can match; detections the
        # tracker can't associate (fast/erratic) still show, just with no id label and no coasting.
        # (ByteTrack's value is on SLOW objects; on fast ones this mode ~= plain nanoowl.)
        now = time.time()
        new_boxes = {}
        synthetic = 0
        for det in dets:
            x0, y0, x1, y1 = [float(v) for v in det.box]
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            tid = self._match_track_id(det.box, tboxes)
            if tid is None:
                synthetic -= 1
                tid = synthetic          # untracked: unique this frame, no id label, no coasting
            prev = self._track_boxes.get(tid) if tid >= 0 else None
            if prev is not None and (now - prev["t"]) > 1e-3:
                vel = np.clip((center - prev["center"]) / (now - prev["t"]), -600.0, 600.0)
            else:
                vel = np.zeros(2)
            new_boxes[tid] = {"box": np.array([x0, y0, x1, y1]), "center": center,
                              "vel": vel, "t": now, "color": _color(tid if tid >= 0 else 0)}
        self._track_boxes = new_boxes   # atomic swap; overlay_live() coasts these on the event loop

        bgr = self._draw_track_boxes(bgr, new_boxes, now)
        pos_ids = sorted(k for k in new_boxes if k >= 0)
        return bgr, (f"Tracking {len(new_boxes)} — IDs {pos_ids}"
                     if new_boxes else f"No matches for {self.prompt}")

    def _draw_track_boxes(self, frame_bgr, state, now):
        """Draw each track's box + #id, coasted by velocity × time-since-inference so the box
        follows the object between inferences (smooth). Modifies frame_bgr in place."""
        h, w = frame_bgr.shape[:2]
        for tid, s in state.items():
            elapsed = now - s["t"]
            if elapsed > 1.0:
                continue
            dx, dy = s["vel"] * elapsed
            x0, y0, x1, y1 = s["box"] + np.array([dx, dy, dx, dy])
            x0i, y0i = int(max(0, x0)), int(max(0, y0))
            x1i, y1i = int(min(w, x1)), int(min(h, y1))
            if x1i - x0i < 2 or y1i - y0i < 2:
                continue
            col = s["color"]
            cv2.rectangle(frame_bgr, (x0i, y0i), (x1i, y1i), col, 2)
            if tid >= 0:                          # only tracked objects get a stable #id label
                cv2.putText(frame_bgr, f"#{tid}", (x0i + 4, max(16, y0i + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        return frame_bgr

    # ---- MODE_TRACK_SAM: OWL → ByteTrack → SAM, mask-only, motion-shifted ----
    def _run_track_sam(self, pil_image, bgr, output, h, w):
        """Segment EVERY OWL detection with SAM (detection parity with the plain SAM mode), and use
        ByteTrack only to lend stable ids + velocity to the detections it can match. Under slow SAM
        the tracker churns and would drop most fast objects, so we must NOT let it gate detection.
        Renders mask-only (colored per track id; no box, no ID text)."""
        import supervision as sv
        dets = output.detections
        # Cap objects/frame: each object costs a SAM decode + a stored full-frame mask that gets
        # re-animated every display frame. 40+ (e.g. a shoal of fish) OOMs/aborts the process.
        # Keep the highest-scoring _MAX_SEG.
        if len(dets) > _MAX_SEG:
            dets = sorted(dets, key=lambda d: (max(d.scores) if d.scores else 0.0),
                          reverse=True)[:_MAX_SEG]
        if dets:
            boxes = np.array([d.box for d in dets], dtype=float)
            scores = np.array([max(d.scores) if d.scores else 0.5 for d in dets], dtype=float)
            class_id = np.array([d.labels[0] if d.labels else 0 for d in dets], dtype=int)
            sv_det = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_id)
        else:
            sv_det = sv.Detections.empty()
        tracked = self._tracker.update_with_detections(sv_det)
        tboxes = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tboxes.append((int(tracked.tracker_id[i]), tracked.xyxy[i]))

        now = time.time()
        try:
            self._sam.set_image(pil_image)
        except Exception as e:
            logger.warning(f"SAM set_image failed: {e}")
            return bgr, "SAM set_image failed"

        new_state = {}
        synthetic = 0                     # negative keys for untracked detections (unique per frame)
        for det in dets:
            x0, y0, x1, y1 = [float(v) for v in det.box]
            x0i, y0i = int(max(0, x0)), int(max(0, y0))
            x1i, y1i = int(min(w, x1)), int(min(h, y1))
            if x1i - x0i < 2 or y1i - y0i < 2:
                continue
            center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            try:
                mask, iou, _ = self._sam.predict(
                    np.array([[x0i, y0i], [x1i, y1i]]), np.array([2, 3]))
            except Exception as e:
                logger.warning(f"SAM predict failed: {e}")
                continue
            mask_bool = self._pick_best_mask(mask, iou, (h, w))
            if mask_bool is None:
                continue
            tid = self._match_track_id(det.box, tboxes)   # borrow a track id if one overlaps
            if tid is None:
                synthetic -= 1
                tid = synthetic           # untracked this frame → no coasting, neutral color
            prev = self._track_masks.get(tid) if tid >= 0 else None
            if prev is not None and (now - prev["t"]) > 1e-3:
                vel = np.clip((center - prev["center"]) / (now - prev["t"]), -600.0, 600.0)
            else:
                vel = np.zeros(2)
            new_state[tid] = {"mask": mask_bool, "center": center, "vel": vel,
                              "t": now, "color": _color(tid if tid >= 0 else 0)}
        self._track_masks = new_state   # atomic swap; overlay_live() reads this on the event loop

        bgr = self._blend_track_masks(bgr, new_state, now)   # render this (inference) frame
        return bgr, (f"Tracking+SAM: {len(new_state)} object(s)"
                     if new_state else f"No matches for {self.prompt}")

    @staticmethod
    def _match_track_id(box, tboxes, iou_thresh=0.3):
        """Return the track id of the best-IoU tracked box (>= iou_thresh), else None."""
        if not tboxes:
            return None
        bx0, by0, bx1, by1 = [float(v) for v in box]
        a1 = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        best_id, best_iou = None, iou_thresh
        for tid, tb in tboxes:
            tx0, ty0, tx1, ty1 = [float(v) for v in tb]
            iw = max(0.0, min(bx1, tx1) - max(bx0, tx0))
            ih = max(0.0, min(by1, ty1) - max(by0, ty0))
            inter = iw * ih
            if inter <= 0:
                continue
            union = a1 + max(0.0, tx1 - tx0) * max(0.0, ty1 - ty0) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, tid
        return best_id

    def _blend_track_masks(self, frame_bgr, state, now):
        """Alpha-blend each track's mask onto frame, shifted by velocity × time-since-inference so
        the mask follows the moving object between SAM runs (shape stale, position tracks)."""
        if not state:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        overlay = frame_bgr.copy()
        drew = False
        for tid, s in state.items():
            elapsed = now - s["t"]
            if elapsed > 1.0:            # too stale to trust the extrapolation — skip
                continue
            m = s["mask"]
            dx, dy = s["vel"] * elapsed
            if abs(dx) >= 1 or abs(dy) >= 1:
                M = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
                m = cv2.warpAffine(m.astype(np.uint8), M, (w, h),
                                   flags=cv2.INTER_NEAREST) > 0
            overlay[m] = s["color"]
            drew = True
        if not drew:
            return frame_bgr
        return cv2.addWeighted(overlay, self.mask_alpha, frame_bgr, 1 - self.mask_alpha, 0)

    # ---- Live-overlay hook (video_processor uses these for MODE_TRACK_SAM) ----
    def wants_live_overlay(self) -> bool:
        """True → overlay onto the LIVE frame every frame (smooth video with coasted boxes /
        motion-shifted masks) instead of freezing on the last baked annotation."""
        return self._mode in (MODE_OWL_TRACK, MODE_TRACK_SAM)

    def overlay_live(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw the current tracked overlay — coasted boxes (OWL_TRACK) or motion-shifted masks
        (TRACK_SAM) — onto the given live frame, extrapolated to 'now'."""
        now = time.time()
        if self._mode == MODE_TRACK_SAM:
            return self._blend_track_masks(frame_bgr, self._track_masks, now)
        if self._mode == MODE_OWL_TRACK:
            return self._draw_track_boxes(frame_bgr.copy(), self._track_boxes, now)
        return frame_bgr

    def _strip_root_detections(self, detections):
        """Remove detections whose label resolves to the implicit root "image" node.
        NanoOWL always emits a whole-frame detection at score 1.0 for the tree root."""
        root_ids = set()
        labels_attr = getattr(self._tree, "labels", None)
        if isinstance(labels_attr, dict):
            for k, v in labels_attr.items():
                if str(v).strip().lower() == "image":
                    try:
                        root_ids.add(int(k))
                    except (TypeError, ValueError):
                        pass
        elif isinstance(labels_attr, (list, tuple)):
            for i, item in enumerate(labels_attr):
                name = item.name if hasattr(item, "name") else str(item)
                if name.strip().lower() == "image":
                    root_ids.add(getattr(item, "id", i))
        if not root_ids:
            return detections
        return [d for d in detections if not d.labels or d.labels[0] not in root_ids]

    def _overlay_sam_masks(self, pil_image, bgr, output, h, w):
        """One set_image per frame, one predict() per detection box, alpha-blend masks."""
        try:
            self._sam.set_image(pil_image)
        except Exception as e:
            logger.warning(f"SAM set_image failed: {e}")
            return bgr
        for det in output.detections:
            x0, y0, x1, y1 = det.box
            x0i, y0i = int(max(0, x0)), int(max(0, y0))
            x1i, y1i = int(min(w, x1)), int(min(h, y1))
            if x1i - x0i < 2 or y1i - y0i < 2:
                continue
            points = np.array([[x0i, y0i], [x1i, y1i]])
            labels = np.array([2, 3])
            try:
                mask, iou, _ = self._sam.predict(points, labels)
            except Exception as e:
                logger.warning(f"SAM predict failed for det {det.id}: {e}")
                continue
            mask_bool = self._pick_best_mask(mask, iou, (h, w))
            if mask_bool is None:
                continue
            overlay = bgr.copy()
            overlay[mask_bool] = _color(det.id)
            bgr = cv2.addWeighted(overlay, self.mask_alpha, bgr, 1 - self.mask_alpha, 0)
        return bgr

    def _run_sam_only(self, pil_image, bgr, h, w):
        """SAM with a full-frame box prompt — diagnostic mode. Without a point/box click
        prompt from the user, SAM almost always returns a huge near-full-frame mask, so
        we relax the degenerate-mask guard for this mode (the mask IS the diagnostic)."""
        if self._sam is None:
            return bgr, "SAM engines not loaded"
        try:
            self._sam.set_image(pil_image)
        except Exception as e:
            return bgr, f"SAM set_image error: {e}"
        points = np.array([[0, 0], [w, h]])
        labels = np.array([2, 3])
        try:
            mask, iou, _ = self._sam.predict(points, labels)
        except Exception as e:
            return bgr, f"SAM predict error: {e}"
        mask_bool = self._pick_best_mask(mask, iou, (h, w), max_coverage=0.99)
        if mask_bool is None:
            return bgr, "SAM returned no usable mask (shape mismatch)"
        overlay = bgr.copy()
        overlay[mask_bool] = _color(0)
        bgr = cv2.addWeighted(overlay, self.mask_alpha, bgr, 1 - self.mask_alpha, 0)
        return bgr, (
            f"SAM-only mask coverage: {100*float(mask_bool.mean()):.0f}%  "
            f"(prompt = full-frame box; click-prompt UI not yet built)"
        )

    @staticmethod
    def _pick_best_mask(mask, iou, expected_shape, max_coverage: float = 0.80):
        """Pick the highest-IoU mask, threshold at 0, reject degenerate (> max_coverage)."""
        n_masks = mask.shape[1] if mask.dim() == 4 else 1
        if n_masks > 1 and iou is not None and iou.numel() >= n_masks:
            best_idx = int(iou[0].argmax().item())
        else:
            best_idx = 0
        mask_bool = (mask[0, best_idx] > 0).detach().cpu().numpy()
        if mask_bool.shape != expected_shape:
            logger.warning(f"Mask shape mismatch: {mask_bool.shape} vs frame {expected_shape}")
            return None
        coverage = float(mask_bool.mean())
        if coverage > max_coverage:
            logger.info(f"Skip degenerate mask: covers {100*coverage:.0f}% of frame")
            return None
        return mask_bool

    def _owl_summary(self, output):
        """Human-readable detection summary. Tree.labels can be list[str], list[obj], or dict —
        build a robust int→name lookup, fall back to str() of the raw label id."""
        n = len(output.detections)
        if n == 0:
            return f"No matches for {self.prompt}"
        label_lookup = {}
        try:
            labels_attr = getattr(self._tree, "labels", None)
            if isinstance(labels_attr, dict):
                label_lookup = {int(k): str(v) for k, v in labels_attr.items()}
            elif isinstance(labels_attr, (list, tuple)):
                for i, item in enumerate(labels_attr):
                    if hasattr(item, "name"):
                        label_lookup[getattr(item, "id", i)] = item.name
                    else:
                        label_lookup[i] = str(item)
        except Exception:
            pass
        parts = []
        for det in output.detections:
            lid = det.labels[0] if det.labels else None
            name = label_lookup.get(lid, str(lid) if lid is not None else "?")
            if det.scores:
                parts.append(f"{name} ({det.scores[0]:.2f})")
            else:
                parts.append(name)
        return f"Detected {n}: " + ", ".join(parts)

    async def process_frame(self, image: PIL.Image.Image, prompt: Optional[str] = None) -> None:
        """Public entrypoint mirroring VLMService.process_frame.
        If `prompt` is provided, switches to it for this and future frames."""
        if self._processing_lock.locked():
            return
        if prompt is not None and prompt != self.prompt:
            self.update_prompt(prompt)

        async with self._processing_lock:
            self.is_processing = True
            t0 = time.perf_counter()
            try:
                # Run the heavy TRT inference off the event loop
                bgr, summary = await asyncio.get_event_loop().run_in_executor(
                    None, self._run_inference, image
                )
                self._last_annotation = bgr
                self.current_response = summary
                dt = time.perf_counter() - t0
                self.last_inference_time = dt
                self.total_inferences += 1
                self.total_inference_time += dt
                logger.info(f"OWL+SAM: {summary} ({dt*1000:.0f}ms)")
            except Exception as e:
                logger.error(f"Inference failed: {type(e).__name__}: {e}", exc_info=True)
                self.current_response = f"Error ({type(e).__name__}): {e}"
            finally:
                self.is_processing = False

    # ---- VLMService-compatible read API ----
    def get_current_response(self) -> Tuple[str, bool]:
        return self.current_response, self.is_processing

    def get_metrics(self) -> dict:
        avg = (self.total_inference_time / self.total_inferences) if self.total_inferences else 0.0
        return {
            "last_latency_ms": self.last_inference_time * 1000,
            "avg_latency_ms": avg * 1000,
            "total_inferences": self.total_inferences,
            "is_processing": self.is_processing,
        }

    def get_last_request_payload(self):
        return self._last_request_payload

    def get_last_response_payload(self):
        return self._last_response_payload

    def get_last_annotation(self) -> Optional[np.ndarray]:
        """BGR ndarray with boxes + masks baked in, or None before first inference."""
        return self._last_annotation

    # ---- Mutation API (called via websocket from index.html) ----
    def update_prompt(self, new_prompt: str, max_tokens: Optional[int] = None) -> None:
        new_prompt = (new_prompt or "").strip()
        if not new_prompt:
            return
        # Be forgiving: NanoOWL needs the tree expression "[a foo, a bar]". If the
        # user typed "a foo, a bar" or just "a foo", wrap it ourselves.
        if not (new_prompt.startswith("[") and new_prompt.endswith("]")):
            new_prompt = "[" + new_prompt.strip("[] ") + "]"
        self.prompt = new_prompt
        # Force re-encode on next inference
        self._encoded_prompt = None
        logger.info(f"Prompt set: {new_prompt}")

    def update_api_settings(self, api_base: Optional[str] = None, api_key: Optional[str] = None) -> None:
        # No remote API; kept for VLMService interface compatibility.
        if api_base:
            self.api_base = api_base
        if api_key is not None:
            self.api_key = api_key if api_key else "N/A"
