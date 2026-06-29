#!/usr/bin/env python3
"""Extend /data/tree_demo/tree_demo.py with NanoSAM mask overlay on top of OWL boxes.
Run AFTER patch_tree_demo.py (which adds --video and the loop). Idempotent: re-running
is a no-op once patched. Run INSIDE the nanoowl container.

The integration: for each frame, call sam_predictor.set_image() ONCE (the expensive
encoder), then sam_predictor.predict() ONCE PER detection box (cheap mask decoder).
Each mask is alpha-blended onto the BGR frame before draw_tree_output adds boxes."""

F = "/data/tree_demo/tree_demo.py"
MARKER = "# >>> SAM PATCH"

s = open(F).read()
if MARKER in s:
    print("already patched:", F)
    raise SystemExit(0)

# --- 1. import nanosam (graceful: degrade to boxes-only if missing) ---
old1 = "from nanoowl.owl_predictor import OwlPredictor\n"
new1 = old1 + (
    "import numpy as np  " + MARKER + " (numpy)\n"
    "try:                                          " + MARKER + " (import)\n"
    "    from nanosam.utils.predictor import Predictor as SamPredictor\n"
    "    _SAM_AVAILABLE = True\n"
    "except Exception as _sam_exc:\n"
    "    SamPredictor = None\n"
    "    _SAM_AVAILABLE = False\n"
    "    print('nanosam not available — running boxes-only:', _sam_exc)\n"
)
assert old1 in s, "anchor 1 (OwlPredictor import) not found"
s = s.replace(old1, new1, 1)

# --- 2. add SAM CLI args (slots in right after --video added by the prior patcher) ---
old2 = '    parser.add_argument("--video", type=str, default="")\n'
new2 = old2 + (
    '    parser.add_argument("--sam_image_encoder_engine", type=str,\n'
    '        default="/data/nanosam/data/resnet18_image_encoder.engine")  ' + MARKER + '\n'
    '    parser.add_argument("--sam_mask_decoder_engine", type=str,\n'
    '        default="/data/nanosam/data/mobile_sam_mask_decoder.engine")  ' + MARKER + '\n'
    '    parser.add_argument("--no_sam", action="store_true",\n'
    '        help="Disable SAM mask overlay (boxes only)")  ' + MARKER + '\n'
    '    parser.add_argument("--mask_alpha", type=float, default=0.5)  ' + MARKER + '\n'
)
assert old2 in s, "anchor 2 (--video arg) not found — run patch_tree_demo.py first"
s = s.replace(old2, new2, 1)

# --- 3. construct SAM predictor + helpers right after the TreePredictor is built ---
old3 = (
    "    predictor = TreePredictor(\n"
    "        owl_predictor=OwlPredictor(\n"
    "            image_encoder_engine=args.image_encode_engine\n"
    "        )\n"
    "    )\n"
)
new3 = old3 + (
    "\n"
    "    " + MARKER + " (predictor + helpers)\n"
    "    sam_predictor = None\n"
    "    if not args.no_sam and _SAM_AVAILABLE:\n"
    "        try:\n"
    "            logging.info('Loading NanoSAM predictor.')\n"
    "            sam_predictor = SamPredictor(\n"
    "                args.sam_image_encoder_engine,\n"
    "                args.sam_mask_decoder_engine,\n"
    "            )\n"
    "        except Exception as _sam_load_exc:\n"
    "            logging.warning(f'NanoSAM failed to load: {_sam_load_exc} — boxes-only.')\n"
    "            sam_predictor = None\n"
    "\n"
    "    _SAM_PALETTE = [\n"
    "        (56, 56, 255), (56, 255, 56), (255, 56, 56), (56, 255, 255),\n"
    "        (255, 56, 255), (255, 255, 56), (255, 128, 0), (128, 0, 255),\n"
    "    ]\n"
    "    def _sam_color(i):\n"
    "        return _SAM_PALETTE[i % len(_SAM_PALETTE)]\n"
    "\n"
    "    def _overlay_sam_masks(bgr, image_pil, tree_output, sam, alpha):\n"
    "        # set_image once per frame (heavy encoder), then one cheap predict() per box.\n"
    "        sam.set_image(image_pil)\n"
    "        h, w = bgr.shape[:2]\n"
    "        for det in tree_output.detections:\n"
    "            x0, y0, x1, y1 = det.box\n"
    "            x0i, y0i = int(max(0, x0)), int(max(0, y0))\n"
    "            x1i, y1i = int(min(w, x1)), int(min(h, y1))\n"
    "            if x1i - x0i < 2 or y1i - y0i < 2:\n"
    "                continue\n"
    "            points = np.array([[x0i, y0i], [x1i, y1i]])\n"
    "            labels = np.array([2, 3])\n"
    "            try:\n"
    "                mask, _iou, _low = sam.predict(points, labels)\n"
    "            except Exception as e:\n"
    "                logging.warning(f'SAM predict failed for det {det.id}: {e}')\n"
    "                continue\n"
    "            mask_bool = (mask[0, 0] > 0).detach().cpu().numpy()\n"
    "            if mask_bool.shape != (h, w):\n"
    "                continue\n"
    "            color = _sam_color(det.id)\n"
    "            overlay = bgr.copy()\n"
    "            overlay[mask_bool] = color\n"
    "            bgr = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)\n"
    "        return bgr\n"
)
assert old3 in s, "anchor 3 (TreePredictor construction) not found"
s = s.replace(old3, new3, 1)

# --- 4. overlay masks BEFORE the existing box-drawing call ---
old4 = "                image = draw_tree_output(image, detections, prompt_data_local['tree'])\n"
new4 = (
    "                if sam_predictor is not None and len(detections.detections) > 0:  " + MARKER + "\n"
    "                    image = _overlay_sam_masks(image, image_pil, detections, sam_predictor, args.mask_alpha)\n"
    "                image = draw_tree_output(image, detections, prompt_data_local['tree'])\n"
)
assert old4 in s, "anchor 4 (draw_tree_output call) not found"
s = s.replace(old4, new4, 1)

open(F, "w").write(s)
print("patched:", F)
