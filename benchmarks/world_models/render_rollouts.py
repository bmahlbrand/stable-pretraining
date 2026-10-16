"""Render probe-decoded rollouts of trained world models as an MP4.

Usage::

    python benchmarks/world_models/render_rollouts.py RUN_DIR [RUN_DIR ...] --out clip.mp4

Each run directory (from ``train.py``) becomes one row: the same held-out
clips, the last given frames, then the rollout. A cross marks the digit
position decoded from the finest level, a circle the coarsest level (HLeWM
MilliVid only), and the label shows the decoded digit class. Needs ``ffmpeg``.
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
from evaluate import Ridge, encode_clips  # noqa: E402
from moving_mnist import MovingMNIST  # noqa: E402
from train import build, given_frames  # noqa: E402
from utils import get_data_dir  # noqa: E402

TITLES = {
    "hlewm": "HLeWM, MilliVid schedule",
    "hlewm-fullres": "HLeWM, full-res schedule",
    "lewm": "LeWM",
}
SCALE, GAP, HEAD, FOOT = 6, 10, 30, 8
SURFACE, TEXT, TEXT2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
FINE, COARSE = (235, 104, 52), (42, 120, 214)


def flat(z):
    return z.reshape(z.shape[0] * z.shape[1], -1)


def decode_run(run_dir: Path, data: dict, clips: list, num_fit: int, horizon: int):
    result = json.loads((run_dir / "results.json").read_text())
    args = argparse.Namespace(**result["args"])
    model = build(args)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    given = given_frames(args)
    video, pos, label = data["video"], data["position"], data["label"]
    levels = encode_clips(model, video[:num_fit])
    onehot = torch.nn.functional.one_hot(label[:num_fit], 10).float()
    onehot = onehot[:, None].expand(-1, video.shape[1], -1)
    size = video.shape[-1]
    with torch.no_grad():
        rollout = model.rollout(video[clips][:, :given], num_frames=given + horizon)
    decoded = []
    for level in sorted({0, len(rollout.embeddings) - 1}):
        z = rollout.embeddings[level]
        pos_probe = Ridge(
            flat(levels[level]), pos[:num_fit].reshape(-1, 2), bounds=(0, size - 1)
        )
        cls_probe = Ridge(flat(levels[level]), onehot.reshape(-1, 10))
        n, t = z.shape[:2]
        decoded.append(
            (
                level,
                pos_probe(flat(z)).reshape(n, t, 2),
                cls_probe(flat(z)).reshape(n, t, 10).argmax(-1),
            )
        )
    return TITLES[args.model], given, decoded


def render_row(title, given, decoded, data, clips, frame, font, small):
    video, label = data["video"], data["label"]
    size = video.shape[-1]
    panel = size * SCALE
    width = len(clips) * panel + (len(clips) + 1) * GAP
    img = Image.new("RGB", (width, HEAD + panel + FOOT), SURFACE)
    d = ImageDraw.Draw(img)
    ahead = frame - given + 1
    status = "given" if ahead <= 0 else f"rollout, {ahead:2d} frame(s) ahead"
    d.text((GAP, 7), f"{title}   frame {frame}   {status}", fill=TEXT, font=font)
    for i, clip in enumerate(clips):
        x0 = GAP + i * (panel + GAP)
        pixels = ((video[clip, frame, 0].numpy() + 1) / 2 * 255).astype(np.uint8)
        tile = Image.fromarray(pixels, "L").resize((panel, panel), Image.NEAREST)
        img.paste(tile.convert("RGB"), (x0, HEAD))
        if ahead <= 0:
            continue
        for level, pos, cls in decoded:
            r, c = pos[i, frame].tolist()
            x, y = x0 + (c + 0.5) * SCALE, HEAD + (r + 0.5) * SCALE
            if level == 0:
                d.line([x - 8, y - 8, x + 8, y + 8], fill=FINE, width=3)
                d.line([x - 8, y + 8, x + 8, y - 8], fill=FINE, width=3)
                guess = int(cls[i, frame])
                mark = "ok" if guess == int(label[clip]) else "wrong"
                d.text(
                    (x0 + 4, HEAD + 4), f"class {guess} ({mark})", fill=FINE, font=small
                )
            else:
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=COARSE, width=3)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, nargs="+", default=[512, 515, 530, 541])
    parser.add_argument("--fit-clips", type=int, default=256)
    parser.add_argument("--show-given", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    first = json.loads((args.runs[0] / "results.json").read_text())["args"]
    given = given_frames(argparse.Namespace(**first))
    data = MovingMNIST(get_data_dir("mnist"), train=False).sample(
        first["eval_clips"], given + args.horizon, torch.Generator().manual_seed(1234)
    )
    rows = [
        decode_run(r, data, args.clips, args.fit_clips, args.horizon) for r in args.runs
    ]
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for frame in range(given - args.show_given, given + args.horizon):
            row_imgs = [
                render_row(t, g, d, data, args.clips, frame, font, small)
                for t, g, d in rows
            ]
            width = max(im.width for im in row_imgs)
            height = sum(im.height for im in row_imgs)
            canvas = Image.new("RGB", (width + width % 2, height + height % 2), SURFACE)
            y = 0
            for im in row_imgs:
                canvas.paste(im, (0, y))
                y += im.height
            repeats = 3 if frame < given else 7
            if frame == given + args.horizon - 1:
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
