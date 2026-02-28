"""Quick test: render a 10-second Ken Burns zoom clip from a single image."""

import subprocess
import sys
from pathlib import Path

from PIL import Image

# Settings matching config.yaml
W, H = 1920, 1080
FPS = 24
RATIO = 0.04
DURATION = 10.0  # seconds

img_path = Path("output/net-worth-levels-where-the-rules-quietly-change/images/001.png")
out_path = Path("output/test_zoom.mp4")

if not img_path.exists():
    print(f"Image not found: {img_path}")
    sys.exit(1)

frames = int(DURATION * FPS)
img = Image.open(img_path).convert("RGB")

print(f"Rendering {frames} frames at {FPS}fps ({DURATION}s) ...")

proc = subprocess.Popen(
    [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:",
        "-c:v", "libx264", "-b:v", "8000k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)

for frame_n in range(frames):
    zoom = 1.0 + RATIO * frame_n / FPS
    inv_z = 1.0 / zoom
    cx = W / 2.0 * (1.0 - inv_z)
    cy = H / 2.0 * (1.0 - inv_z)
    frame = img.transform(
        (W, H),
        Image.AFFINE,
        data=(inv_z, 0, cx, 0, inv_z, cy),
        resample=Image.BICUBIC,
    )
    proc.stdin.write(frame.tobytes())
    if (frame_n + 1) % FPS == 0:
        print(f"  {frame_n + 1}/{frames} frames")

proc.stdin.close()
proc.wait()

if proc.returncode != 0:
    print(f"ffmpeg error: {proc.stderr.read().decode()[-500:]}")
    sys.exit(1)

print(f"Done! Output: {out_path}")
