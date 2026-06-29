#!/usr/bin/env python3
"""Make a persistent, video-capable copy of NanoOWL's tree_demo.
Run INSIDE the nanoowl container. Copies examples/tree_demo -> /data/tree_demo and
patches it to (a) accept --video <path> and (b) loop the video forever (seek to 0 on EOF).
Idempotent: safe to re-run."""
import os, shutil

SRC = "/opt/nanoowl/examples/tree_demo"
DST = "/data/tree_demo"

if not os.path.exists(DST):
    shutil.copytree(SRC, DST)
    print("copied", SRC, "->", DST)

f = os.path.join(DST, "tree_demo.py")
s = open(f).read()

changed = False

# 1) add --video arg + use it
if "--video" not in s:
    a = '    parser.add_argument("--camera", type=int, default=0)'
    assert a in s, "camera arg not found — tree_demo.py layout changed"
    s = s.replace(a, a + '\n    parser.add_argument("--video", type=str, default="")')
    s = s.replace("CAMERA_DEVICE = args.camera",
                  "CAMERA_DEVICE = args.video if args.video else args.camera")
    changed = True

# 2) loop on EOF. Match only the if/return pair (the file has blank lines between
#    statements, so matching the camera.read() block as well is fragile).
old = "            if not re:\n                return re, None"
new = ("            if not re:\n"
       "                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)\n"
       "                re, image = camera.read()\n"
       "                if not re:\n"
       "                    return re, None")
if "CAP_PROP_POS_FRAMES" not in s:
    assert old in s, "read-fail block not found — tree_demo.py layout changed"
    s = s.replace(old, new)
    changed = True

open(f, "w").write(s)
print("patched" if changed else "already patched", f)
