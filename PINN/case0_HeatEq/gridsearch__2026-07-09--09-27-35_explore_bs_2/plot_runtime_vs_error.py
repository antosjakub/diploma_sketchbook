import json
import os
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(CURRENT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt


OUTPUT_PATH_LINF = CURRENT_DIR / "scatter_linf.png"
OUTPUT_PATH_REL_L2 = CURRENT_DIR / "scatter_rel_l2.png"
MARKER_SIZE = 90
ANNOTATION_SIZE = 13


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_runs(base_dir: Path):
    grouped = {
        '13': defaultdict(
            lambda: {
                "runtime": [],
                "linf_pde": [],
                "linf_ic": [],
                "rel_l2_pde": [],
                "rel_l2_ic": [],
            }
        ),
        '42': defaultdict(
            lambda: {
                "runtime": [],
                "linf_pde": [],
                "linf_ic": [],
                "rel_l2_pde": [],
                "rel_l2_ic": [],
            }
        ),
    }

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        report_path = run_dir / "report.json"
        metadata_path = run_dir / "model_metadata.json"
        if not report_path.exists() or not metadata_path.exists():
            continue

        report = load_json(report_path)
        metadata = load_json(metadata_path)
        args = metadata.get("args", {})

        seed = str(args.get("seed"))

        bs = args.get("bs")
        if bs is None:
            continue
        
        grouped[seed][bs]["runtime"].append(report["runtime"])
        grouped[seed][bs]["linf_pde"].append(report["test_linf"]["pde"])
        grouped[seed][bs]["linf_ic"].append(report["test_linf"]["ic"])
        grouped[seed][bs]["rel_l2_pde"].append(report["test_rel_l2"]["pde"])
        grouped[seed][bs]["rel_l2_ic"].append(report["test_rel_l2"]["ic"])

    return grouped


def average(values):
    return sum(values) / len(values)


def plot_series(ax, grouped, metric_key, color, label, annotation_prefix):
    batch_sizes = sorted(grouped)
    runtimes = [average(grouped[bs]["runtime"]) for bs in batch_sizes]
    metric_values = [average(grouped[bs][metric_key]) for bs in batch_sizes]

    ax.plot(runtimes, metric_values, color=color, linewidth=1.8, label=label, zorder=1)
    ax.scatter(runtimes, metric_values, s=MARKER_SIZE, color=color, zorder=2)

    for runtime, metric_value, bs in zip(runtimes, metric_values, batch_sizes):
        ax.annotate(
            f"{annotation_prefix}{bs}",
            (runtime, metric_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            color=color,
        )


def plot_linf(grouped):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    panels = (
        (axes[0], "linf_pde", r"$L_\infty$ Error (PDE)"),
        (axes[1], "linf_ic", r"$L_\infty$ Error (IC)"),
    )
    colors = {"13": "C0", "42": "C1"}

    for ax, metric_key, title in panels:
        for seed, seed_grouped in grouped.items():
            plot_series(
                ax,
                seed_grouped,
                metric_key,
                color=colors.get(seed, "C2"),
                label=f"seed={seed}",
                annotation_prefix="bs=",
            )

        ax.set_xlabel("Runtime [s]")
        ax.set_ylabel(r"$L_\infty$ error")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH_LINF, dpi=300)
    plt.close(fig)


def plot_rel_l2(grouped):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    panels = (
        (axes[0], "rel_l2_pde", "Relative $L_2$ Error (PDE)"),
        (axes[1], "rel_l2_ic", "Relative $L_2$ Error (IC)"),
    )
    colors = {"13": "C0", "42": "C1"}

    for ax, metric_key, title in panels:
        for seed, seed_grouped in grouped.items():
            plot_series(
                ax,
                seed_grouped,
                metric_key,
                color=colors.get(seed, "C2"),
                label=f"seed={seed}",
                annotation_prefix="bs=",
            )

        ax.set_xlabel("Runtime [s]")
        ax.set_ylabel(r"Relative $L_2$ error")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH_REL_L2, dpi=300)
    plt.close(fig)


def main():
    grouped = collect_runs(CURRENT_DIR)
    plot_linf(grouped)
    plot_rel_l2(grouped)


if __name__ == "__main__":
    main()
