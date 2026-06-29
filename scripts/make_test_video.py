#!/usr/bin/env python3
"""Build a looping test video from NanoOWL's own sample images.
Run INSIDE the nanoowl container. Writes /data/test.mp4 (persists on the host).
Lets us demo with no camera attached. OWL-ViT detects real objects in these photos,
so prompts like '[an owl, a glove, a frog]' will draw boxes."""
import cv2, glob, os

ASSETS = "/opt/nanoowl/assets/*.jpg"
OUT = "/data/test.mp4"
W, H, FPS = 640, 480, 15
# The demo runs detection as fast as it can and ignores the video's FPS, so an image's
# on-screen dwell = FRAMES_PER_IMG / inference_fps. Raise this if images switch too fast.
FRAMES_PER_IMG = 150

imgs = [p for p in sorted(glob.glob(ASSETS)) if "out" not in os.path.basename(p)]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for p in imgs:
    im = cv2.imread(p)
    if im is None:
        continue
    im = cv2.resize(im, (W, H))
    for _ in range(FRAMES_PER_IMG):
        vw.write(im)
vw.release()
print("images used:", imgs)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
