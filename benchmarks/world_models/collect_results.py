"""Aggregate world-model benchmark runs into a markdown table and a chart.

Usage::

    python benchmarks/world_models/collect_results.py RUNS_DIR [--plot out.png]

Reads every ``RUNS_DIR/*/results.json`` written by ``train.py``, groups runs by
model, and reports the mean and standard deviation over seeds.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HORIZONS = (1, 4, 8, 16, 32)
LABELS = {
    "lewm": "LeWM",
    "hlewm-fullres": "HLeWM, full-res schedule",
    "hlewm": "HLeWM, MilliVid schedule",
}
METRICS = {
    "iou": "Occupancy IoU of decoded rollouts (higher is better)",
    "reemerged_recall": "Recall on re-emerging digits (higher is better)",
    "false_pos": "False positives on empty cells (lower is better)",
}


def load(runs_dir: Path) -> dict:
    runs = defaultdict(list)
    for path in sorted(runs_dir.glob("*/results.json")):
        result = json.loads(path.read_text())
        runs[result["args"]["model"]].append(result)
    return runs


def fmt(values) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.nanmean(values):.3f} ± {np.nanstd(values, ddof=1):.3f}"


def curves(runs, level, metric, source="rollout"):
    """Per-seed curves ``[seeds, horizon]``.

    ``source`` is ``"rollout"``, ``"rollout_mismatched"`` or a reference name.
    """
    if source.startswith("rollout"):
        return np.array([r["eval"][source][level][metric] for r in runs], dtype=float)
    return np.array([r["eval"]["reference"][source][metric] for r in runs], dtype=float)


def num_levels(runs):
    return len(runs[0]["eval"]["rollout"]) if runs else 0


def table(runs: dict) -> str:
    lines = []
    head = "| Model | Level | Seeds | " + " | ".join(f"h={h}" for h in HORIZONS)
    for metric, title in METRICS.items():
        lines += [f"### {title}", "", head + " | Mean 1-32 |"]
        lines.append("|" + "---|" * (len(HORIZONS) + 4))
        for model in LABELS:
            rs = runs.get(model, [])
            for level in range(num_levels(rs)):
                c = curves(rs, level, metric)
                cells = [fmt(c[:, h - 1]) for h in HORIZONS]
                lines.append(
                    f"| {LABELS[model]} | {level} | {len(c)} | "
                    + " | ".join(cells)
                    + f" | {fmt(np.nanmean(c, 1))} |"
                )
            if rs:
                c = curves(rs, 0, metric, "copy_last_frame")
                cells = [fmt(c[:, h - 1]) for h in HORIZONS]
                lines.append(
                    f"| {LABELS[model]}: copy last frame | 0 | {len(c)} | "
                    + " | ".join(cells)
                    + f" | {fmt(np.nanmean(c, 1))} |"
                )
        lines.append("")

    lines += [
        "### Memory: recall on re-emerging digits, mean over horizons 1-32",
        "",
        "Scored against the clip's own truth and against another clip's truth;",
        "the difference is what the model remembers about this clip's hidden",
        "digit rather than a generic guess near the occluder.",
        "",
        "| Model | Level | Own clip | Other clip | Clip-specific |",
        "|---|---|---|---|---|",
    ]
    for model in LABELS:
        rs = [r for r in runs.get(model, []) if "rollout_mismatched" in r["eval"]]
        for level in range(num_levels(rs)):
            own = np.nanmean(curves(rs, level, "reemerged_recall"), 1)
            other = np.nanmean(
                curves(rs, level, "reemerged_recall", "rollout_mismatched"), 1
            )
            lines.append(
                f"| {LABELS[model]} | {level} | {fmt(own)} | {fmt(other)} | "
                f"{fmt(own - other)} |"
            )
    any_run = next(iter(runs.values()))[0]["eval"]["reference"]
    if "beside_occluder" in any_run:
        guess = np.nanmean(any_run["beside_occluder"]["reemerged_recall"])
        lines.append(
            f"| Fixed guess: cells beside the occluder | - | {guess:.3f} | - | - |"
        )
    lines.append("")

    lines += [
        "### Probes on the true future frames, per level (finest first)",
        "",
        "| Model | IoU | Re-emerged recall | Train time (min) |",
        "|---|---|---|---|",
    ]
    for model in LABELS:
        rs = runs.get(model, [])
        if not rs:
            continue

        def per_level(key, rs=rs):
            vals = np.array([[lv[key] for lv in r["eval"]["true_frames"]] for r in rs])
            return ", ".join(f"{v:.3f}" for v in vals.mean(0))

        minutes = np.mean([r["train"]["seconds"] for r in rs]) / 60
        lines.append(
            f"| {LABELS[model]} | {per_level('iou')} | "
            f"{per_level('reemerged_recall')} | {minutes:.0f} |"
        )
    return "\n".join(lines)


def plot(runs: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, text2, muted, grid = (
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#8a8985",
        "#e6e5e1",
    )
    lines = [
        ("lewm", 0, "LeWM", "#1baf7a"),
        ("hlewm-fullres", 0, "HLeWM full-res schedule", "#6250d6"),
        ("hlewm", 0, "HLeWM MilliVid, fine level", "#eb6834"),
        ("hlewm", 2, "HLeWM MilliVid, coarse level", "#2a78d6"),
    ]
    panels = (
        ("iou", "Occupancy IoU"),
        ("reemerged_recall", "Recall on re-emerging digits"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    for ax, (metric, ylabel) in zip(axes, panels):
        ax.set_facecolor(surface)
        for model, level, label, color in lines:
            rs = runs.get(model, [])
            if level >= num_levels(rs):
                continue
            c = curves(rs, level, metric)
            h = np.arange(1, c.shape[1] + 1)
            ax.plot(h, np.nanmean(c, 0), color=color, lw=2.2, label=label)
            if len(c) > 1:
                ax.fill_between(
                    h, np.nanmin(c, 0), np.nanmax(c, 0), color=color, alpha=0.15, lw=0
                )
        ref = runs.get("hlewm") or next(iter(runs.values()))
        c = curves(ref, 0, metric, "copy_last_frame")
        h = np.arange(1, c.shape[1] + 1)
        ax.plot(
            h, np.nanmean(c, 0), color=muted, ls="--", lw=1.4, label="Copy last frame"
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(1, h[-1])
        ax.set_xlabel("Frames past the last given frame", color=text2)
        ax.set_ylabel(ylabel, color=text2)
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=text2)
        ax.legend(frameon=False, fontsize=8, labelcolor=text)
    fig.suptitle(
        "Occluded 4-digit Moving MNIST: probe-decoded latent rollouts "
        "(mean over seeds, band = min-max)",
        color=text,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    runs = load(args.runs_dir)
    if not runs:
        raise SystemExit(f"No results.json under {args.runs_dir}")
    print(table(runs))
    if args.plot:
        plot(runs, args.plot)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
