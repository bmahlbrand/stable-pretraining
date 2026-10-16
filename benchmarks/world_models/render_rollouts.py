"""Render probe-decoded rollouts of trained world models as an MP4.

Usage::

    python benchmarks/world_models/render_rollouts.py RUN_DIR [RUN_DIR ...] --out clip.mp4

Each run directory (from ``train.py``) becomes one row showing the same
held-out test clips (each has a digit that re-emerges from behind the band
during the rollout): the last given frames, then the rollout. Outlined cells
are where the model's finest-level rollout, decoded by an occupancy probe,
places a digit. Needs ``ffmpeg``.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import OccupancyProbe, encode_clips  # noqa: E402
from moving_mnist import IGNORE  # noqa: E402
from train import PATCH, build, eval_clips, given_frames  # noqa: E402
from utils import get_data_dir  # noqa: E402

TITLES = {
    "hlewm": "HLeWM, MilliVid schedule",
    "hlewm-fullres": "HLeWM, full-res schedule",
    "lewm": "LeWM",
}
SCALE, GAP, HEAD, FOOT = 3, 10, 30, 8
SURFACE, TEXT = (252, 252, 251), (11, 11, 11)
PRED = (235, 104, 52)


def flat(z):
    return z.reshape(z.shape[0] * z.shape[1], -1)


def decode_run(run_dir: Path, data: dict, clips: list, num_fit: int, horizon: int):
    result = json.loads((run_dir / "results.json").read_text())
    args = argparse.Namespace(**result["args"])
    model = build(args)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    given = given_frames(args)
    fine = encode_clips(model, data["video"][:num_fit])[0]
    owner = data["owner"][:num_fit].flatten(2)
    probe = OccupancyProbe(flat(fine[:, ::2]), flat(owner[:, ::2]))
    with torch.no_grad():
        rollout = model.rollout(
            data["video"][clips][:, :given], num_frames=given + horizon
        )
    z = rollout.embeddings[0]
    occupied = probe(flat(z)).reshape(*z.shape[:2], *data["owner"].shape[-2:])
    # Cells under the band are never scored or trained on; don't draw them.
    occupied &= data["owner"][0, 0] != IGNORE
    return TITLES[args.model], given, occupied


def render_row(title, given, occupied, video, frame, font):
    n, size = occupied.shape[0], video.shape[-1]
    panel, cell = size * SCALE, PATCH * SCALE
    width = n * panel + (n + 1) * GAP
    img = Image.new("RGB", (width, HEAD + panel + FOOT), SURFACE)
    d = ImageDraw.Draw(img)
    ahead = frame - given + 1
    status = "given" if ahead <= 0 else f"rollout, {ahead:2d} frame(s) ahead"
    d.text((GAP, 7), f"{title}   frame {frame}   {status}", fill=TEXT, font=font)
    for i in range(n):
        x0 = GAP + i * (panel + GAP)
        pixels = ((video[i, frame, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
        tile = Image.fromarray(pixels, "L").resize((panel, panel), Image.NEAREST)
        img.paste(tile.convert("RGB"), (x0, HEAD))
        if ahead <= 0:
            continue
        for r, c in occupied[i, frame].nonzero().tolist():
            x, y = x0 + c * cell, HEAD + r * cell
            d.rectangle(
                [x + 1, y + 1, x + cell - 2, y + cell - 2], outline=PRED, width=2
            )
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--fit-clips", type=int, default=256)
    parser.add_argument("--show-given", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    first = argparse.Namespace(
        **json.loads((args.runs[0] / "results.json").read_text())["args"]
    )
    data = eval_clips(get_data_dir("mnist"), given_frames(first), first)
    clips = [first.fit_clips + c for c in args.clips]
    horizon = first.horizon
    rows = [decode_run(r, data, clips, args.fit_clips, horizon) for r in args.runs]
    video = data["video"][clips]
    font = ImageFont.load_default(size=15)
    given = rows[0][1]
    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for frame in range(given - args.show_given, given + horizon):
            row_imgs = [render_row(t, g, o, video, frame, font) for t, g, o in rows]
            width = max(im.width for im in row_imgs)
            height = sum(im.height for im in row_imgs)
            canvas = Image.new("RGB", (width + width % 2, height + height % 2), SURFACE)
            y = 0
            for im in row_imgs:
                canvas.paste(im, (0, y))
                y += im.height
            repeats = 3 if frame < given else 7
            if frame == given + horizon - 1:
                repeats += 50
            for _ in range(repeats):
                canvas.save(f"{tmp}/{k:05d}.png")
                k += 1
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i",
             f"{tmp}/%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-movflags", "+faststart", str(args.out)],
            check=True,
        )  # fmt: skip
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
