"""Assemble the Part 2 demo video: title cards + result clips, letterboxed to 1280x720.

  python scripts/make_demo_video.py --out results/part2_demo.mp4
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 20


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def card(title, lines, seconds=4.0):
    img = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(img)
    d.text((80, 200), title, fill=(11, 11, 11), font=font(40, True))
    y = 290
    for ln in lines:
        d.text((80, y), ln, fill=(82, 81, 78), font=font(24))
        y += 40
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def fit(frame):
    img = Image.fromarray(frame)
    s = min(W / img.width, H / img.height)
    img = img.resize((int(img.width * s) // 2 * 2, int(img.height * s) // 2 * 2))
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return np.asarray(canvas)


def clip(path, caption=None, max_s=None):
    r = imageio.get_reader(path)
    src_fps = r.get_meta_data().get("fps", FPS)
    frames = []
    for i, f in enumerate(r):
        t = i / src_fps
        if max_s and t > max_s:
            break
        while len(frames) < int(t * FPS) + 1:          # resample to the output fps
            fr = fit(f)
            if caption:
                img = Image.fromarray(fr)
                d = ImageDraw.Draw(img)
                d.rectangle([0, H - 44, W, H], fill=(252, 252, 251))
                d.text((20, H - 36), caption, fill=(11, 11, 11), font=font(20))
                fr = np.asarray(img)
            frames.append(fr)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "part2_demo.mp4"))
    ap.add_argument("--spec", default=str(ROOT / "scripts" / "demo_video_spec.txt"),
                    help="lines: 'card|Title|line1;line2' or 'clip|path|caption|max_seconds'")
    a = ap.parse_args()
    frames = []
    for line in Path(a.spec).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, *rest = line.split("|")
        if kind == "card":
            frames += card(rest[0], rest[1].split(";") if len(rest) > 1 else [],
                           float(rest[2]) if len(rest) > 2 else 4.0)
        else:
            p = ROOT / rest[0]
            if not p.exists():
                raise FileNotFoundError(f"Required demo clip not found: {p}")
            frames += clip(p, rest[1] if len(rest) > 1 else None, float(rest[2]) if len(rest) > 2 and rest[2] else None)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "raw.mp4"
        w = imageio.get_writer(tmp, fps=FPS, quality=7, macro_block_size=8)
        for f in frames:
            w.append_data(f)
        w.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "26",
                        "-preset", "slow", "-pix_fmt", "yuv420p", a.out], check=True)
    print("wrote", a.out, f"{len(frames) / FPS:.0f}s")


if __name__ == "__main__":
    main()
