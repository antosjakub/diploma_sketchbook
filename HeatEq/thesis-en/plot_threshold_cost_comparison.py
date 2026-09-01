from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
})


TARGET_REL_L2 = 0.0025
OUTPUT_PATH = Path(__file__).resolve().parent / "figures" / "threshold_cost_comparison_rel_l2.png"


@dataclass(frozen=True)
class SeriesPoint:
    dimension: int
    dof: int
    runtime_seconds: float | None
    label: str


def pinn_parameter_count(dimension: int, hidden_layers: int, width: int, output_dim: int = 1) -> int:
    """Trainable parameters for a fully connected MLP with uniform hidden width."""
    input_dim = dimension + 1  # space variables + time

    total = input_dim * width + width
    total += (hidden_layers - 1) * (width * width + width)
    total += width * output_dim + output_dim
    return total


def minutes(value: float) -> float:
    return 60.0 * value


def main() -> None:
    # Sparse-grid points selected as the first/cheapest runs with rel L2 <= 0.0025.
    sg_points = [
        SeriesPoint(dimension=2, dof=577, runtime_seconds=0.2, label="n=6"),
        SeriesPoint(dimension=4, dof=133_889, runtime_seconds=71.14, label="n=8"),
        SeriesPoint(dimension=6, dof=10_000_000, runtime_seconds=minutes(45), label="n=9"),
    ]

    # PINN points selected as the cheapest reported runs with rel L2 <= 0.0025.
    # The 2D runtime is not included because the thesis text only states "within a few minutes".
    pinn_points = [
        SeriesPoint(
            dimension=2,
            dof=pinn_parameter_count(dimension=2, hidden_layers=4, width=64),
            runtime_seconds=minutes(2.5),
            label="4x64, 20k",
        ),
        SeriesPoint(
            dimension=4,
            dof=pinn_parameter_count(dimension=4, hidden_layers=4, width=128),
            runtime_seconds=minutes(9),
            label="4x128, 40k",
        ),
        SeriesPoint(
            dimension=6,
            dof=pinn_parameter_count(dimension=6, hidden_layers=4, width=256),
            runtime_seconds=minutes(30),
            label="4x256, 80k",
        ),
        SeriesPoint(
            dimension=8,
            dof=pinn_parameter_count(dimension=8, hidden_layers=4, width=512),
            runtime_seconds=minutes(55),
            label="4x512, 80k",
        ),
    ]

    fig, (ax_dof, ax_runtime) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    plot_series(
        ax=ax_dof,
        sg_points=sg_points,
        pinn_points=pinn_points,
        y_attr="dof",
        ylabel="degrees of freedom",
        title=f"runs that reached rel $L^2 \\leq {TARGET_REL_L2}$",
    )
    ax_dof.set_yscale("log")

    plot_series(
        ax=ax_runtime,
        sg_points=sg_points,
        pinn_points=pinn_points,
        y_attr="runtime_seconds",
        ylabel="tuntime [s]",
        title="runtime of thus selected runs",
    )
    ax_runtime.set_yscale("log")

    #note = (
    #    "PINN DOF = trainable parameters of a 4-hidden-layer FFNN. "
    #    "Combination techniq."
    #)
    #fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")
def plot_series(
    ax: plt.Axes,
    sg_points: list[SeriesPoint],
    pinn_points: list[SeriesPoint],
    y_attr: str,
    ylabel: str,
    title: str,
) -> None:
    label_offsets = {
        ("sg", "dof", 2): (6, 6),
        ("sg", "dof", 4): (6, 6),
        ("sg", "dof", 6): (6, 6),
        ("sg", "runtime_seconds", 2): (6, 6),
        ("sg", "runtime_seconds", 4): (6, 6),
        ("sg", "runtime_seconds", 6): (6, 8),
        ("pinn", "dof", 2): (6, -14),
        ("pinn", "dof", 4): (6, -14),
        ("pinn", "dof", 6): (6, -14),
        ("pinn", "dof", 8): (6, -14),
        ("pinn", "runtime_seconds", 2): (6, -14),
        ("pinn", "runtime_seconds", 4): (6, -14),
        ("pinn", "runtime_seconds", 6): (6, -14),
        ("pinn", "runtime_seconds", 8): (6, -14),
    }

    label_box = {
        "boxstyle": "round,pad=0.15",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.9,
    }

    sg_x = [point.dimension for point in sg_points if getattr(point, y_attr) is not None]
    sg_y = [getattr(point, y_attr) for point in sg_points if getattr(point, y_attr) is not None]
    pinn_x = [point.dimension for point in pinn_points if getattr(point, y_attr) is not None]
    pinn_y = [getattr(point, y_attr) for point in pinn_points if getattr(point, y_attr) is not None]

    ax.plot(sg_x, sg_y, marker="o", linewidth=2, label="combination technique")
    ax.plot(pinn_x, pinn_y, marker="s", linewidth=2, label="PINN")

    for point in sg_points:
        value = getattr(point, y_attr)
        if value is not None:
            offset = label_offsets.get(("sg", y_attr, point.dimension), (6, 6))
            ax.annotate(
                point.label,
                (point.dimension, value),
                xytext=offset,
                textcoords="offset points",
                fontsize=13,
                bbox=label_box,
            )

    for point in pinn_points:
        value = getattr(point, y_attr)
        if value is not None:
            offset = label_offsets.get(("pinn", y_attr, point.dimension), (6, -14))
            ax.annotate(
                point.label,
                (point.dimension, value),
                xytext=offset,
                textcoords="offset points",
                fontsize=13,
                bbox=label_box,
            )

    ax.set_xlabel("dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({point.dimension for point in sg_points + pinn_points}))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


if __name__ == "__main__":
    main()
