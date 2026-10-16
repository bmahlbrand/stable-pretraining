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


def load(runs_dir: Path) -> dict:
    runs = defaultdict(list)
    for path in sorted(runs_dir.glob("*/results.json")):
        result = json.loads(path.read_text())
        runs[result["args"]["model"]].append(result)
    return runs


def mean_std(values) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def series(runs, metric, level):
    """Per-seed curves ``[seeds, horizon]`` for one level of a rollout metric."""
    return np.array([r["eval"][metric][level] for r in runs])


def table(runs: dict) -> str:
    head = "| Model | Level | Seeds | " + " | ".join(f"h={h}" for h in HORIZONS)
    lines = [
        "### Position error of decoded rollouts (px, lower is better)",
        "",
        head + " | Mean 1-32 |",
        "|" + "---|" * (len(HORIZONS) + 4),
    ]
    for model in LABELS:
        for level in _levels(runs.get(model, [])):
            curves = series(runs[model], "rollout_position_px", level)
            cells = [mean_std(curves[:, h - 1]) for h in HORIZONS]
            lines.append(
                f"| {LABELS[model]} | {level} | {len(curves)} | "
                + " | ".join(cells)
                + f" | {mean_std(curves.mean(1))} |"
            )
    ref = next(iter(runs.values()))[0]["eval"]["reference_position_px"]
    for name in ("constant_velocity_no_walls", "mean_position"):
        cells = [f"{ref[name][h - 1]:.2f}" for h in HORIZONS]
        lines.append(
            f"| {name} | - | - | "
            + " | ".join(cells)
            + f" | {np.mean(ref[name]):.2f} |"
        )

    lines += [
        "",
        "### Digit-class accuracy of decoded rollouts (higher is better)",
        "",
        head + " | Mean 1-32 |",
        "|" + "---|" * (len(HORIZONS) + 4),
    ]
    for model in LABELS:
        for level in _levels(runs.get(model, [])):
            curves = series(runs[model], "rollout_class_acc", level)
            cells = [mean_std(curves[:, h - 1]) for h in HORIZONS]
            lines.append(
                f"| {LABELS[model]} | {level} | {len(curves)} | "
                + " | ".join(cells)
                + f" | {mean_std(curves.mean(1))} |"
            )

    lines += [
        "",
        "### Encoder probes on true frames, per level (finest first)",
        "",
        "| Model | Position error (px) | Class accuracy | Copy-prev MSE / var |"
        " Copy-last position error, h=1 / h=16 (px) | Train time (min) |",
        "|---|---|---|---|---|---|",
    ]
    for model in LABELS:
        rs = runs.get(model, [])
        if not rs:
            continue

        def per_level(key, rs=rs):
            vals = np.array([r["eval"][key] for r in rs])
            return ", ".join(f"{v:.2f}" for v in vals.mean(0))

        copy = np.array(
            [r["eval"]["reference_position_px"]["copy_last_frame"] for r in rs]
        )
        minutes = np.mean([r["train"]["seconds"] for r in rs]) / 60
        lines.append(
            f"| {LABELS[model]} | {per_level('probe_position_px')} | "
            f"{per_level('probe_class_acc')} | {per_level('smoothness')} | "
            f"{copy[:, 0].mean():.2f} / {copy[:, 15].mean():.2f} | {minutes:.0f} |"
        )
    return "\n".join(lines)


def _levels(runs):
    if not runs:
        return []
    return range(len(runs[0]["eval"]["rollout_position_px"]))


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
        ("lewm", 0, "LeWM", "#1baf7a", "-"),
        ("hlewm-fullres", 0, "HLeWM full-res schedule", "#6250d6", "-"),
        ("hlewm", 0, "HLeWM MilliVid, fine level", "#eb6834", "-"),
        ("hlewm", 2, "HLeWM MilliVid, coarse level", "#2a78d6", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    for ax, metric, ylabel in (
        (axes[0], "rollout_position_px", "Position error (px, 32×32 frame)"),
        (axes[1], "rollout_class_acc", "Digit-class accuracy"),
    ):
        ax.set_facecolor(surface)
        top = 0.0
        for model, level, label, color, ls in lines:
            if model not in runs or level not in _levels(runs[model]):
                continue
            curves = series(runs[model], metric, level)
            h = np.arange(1, curves.shape[1] + 1)
            top = max(top, curves.max())
            ax.plot(h, curves.mean(0), color=color, ls=ls, lw=2.2, label=label)
            if len(curves) > 1:
                ax.fill_between(
                    h, curves.min(0), curves.max(0), color=color, alpha=0.15, lw=0
                )
        ref = next(iter(runs.values()))[0]["eval"]
        if metric == "rollout_position_px":
            refs = ref["reference_position_px"]
            ax.plot(
                h,
                refs["mean_position"],
                color=muted,
                ls="--",
                lw=1.4,
                label="Mean position",
            )
            ax.plot(
                h,
                refs["constant_velocity_no_walls"],
                color=muted,
                ls=":",
                lw=1.4,
                label="Constant velocity, no walls",
            )
            top = max(top, max(refs["mean_position"]))
            ax.set_ylim(0, 1.1 * top)
        else:
            ax.axhline(0.1, color=muted, ls="--", lw=1.4, label="Chance")
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
        "Moving MNIST: probe-decoded latent rollouts (mean over seeds, band = min-max)",
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
