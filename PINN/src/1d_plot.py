"""
thesis_plot.py - Minimal plotting helper for a thesis.

    from thesis_plot import ThesisPlot

    tp = ThesisPlot(title="E vs t", xlabel="$t$ [s]", ylabel="$E$ [J]", logy=True)
    tp.plot(x, y, label="data", marker="o", linestyle="none")
    tp.plot(x_fit, y_fit, label="fit")
    tp.legend()
    tp.save("energy.pdf")

Torch tensors are converted to numpy automatically.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _np(v):
    """Torch tensor / list → numpy array."""
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


# Shared rc overrides applied once on import – keeps every figure consistent.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})


class ThesisPlot:
    """Thin wrapper around a single matplotlib Figure + Axes."""

    def __init__(
        self,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        *,
        logy: bool = False,
        logx: bool = False,
        figsize: tuple[float, float] = (5.5, 4.0),
        dpi: int = 150,
    ):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        if title:
            self.ax.set_title(title)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if logx:
            self.ax.set_xscale("log")
        if logy:
            self.ax.set_yscale("log")

    # ── plotting ─────────────────────────────────────────────────────

    def add_plot(self, x, y, **kwargs):
        """Line / scatter / any combo – delegates straight to ax.plot."""
        self.ax.plot(_np(x), _np(y), **kwargs)
        return self

    def errorbar(self, x, y, *, yerr=None, xerr=None, **kwargs):
        kwargs.setdefault("fmt", "o")
        kwargs.setdefault("capsize", 2.5)
        self.ax.errorbar(
            _np(x), _np(y),
            yerr=_np(yerr) if yerr is not None else None,
            xerr=_np(xerr) if xerr is not None else None,
            **kwargs,
        )
        return self

    def fill_between(self, x, y_lo, y_hi, **kwargs):
        kwargs.setdefault("alpha", 0.2)
        self.ax.fill_between(_np(x), _np(y_lo), _np(y_hi), **kwargs)
        return self

    # ── helpers ──────────────────────────────────────────────────────

    def legend(self, **kwargs):
        self.ax.legend(**kwargs)
        return self

    def integer_xticks(self, step: int = 1):
        """Force x-axis ticks to be integers, optionally every `step`."""
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if step > 1:
            lo, hi = self.ax.get_xlim()
            self.ax.set_xticks(range(int(lo) + 1, int(hi) + 1, step))
        return self

    def xlim(self, lo=None, hi=None):
        self.ax.set_xlim(lo, hi)
        return self

    def ylim(self, lo=None, hi=None):
        self.ax.set_ylim(lo, hi)
        return self

    def save(self, path: str | Path, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(str(path), **kwargs)
        return self

    def show(self):
        plt.show()
        return self

    def close(self):
        plt.close(self.fig)


if __name__ == "__main__":
    import torch
    max_d = 20
    d = torch.linspace(1,max_d, max_d, dtype=int)
    y = (1/(2*torch.pi))**(d/2)

    plt.scatter(d,y)
    plt.yscale("log")
    plt.xticks(range(1,21,1))
    plt.savefig("max_val_gauss.png")

    #tp = ThesisPlot(
    #    title="max value of unit gaussian with increasing dimension (d)", xlabel="$d$", ylabel="$f(0)$", logy=True,
    #    figsize=(6,3)
    #)
    #tp.add_plot(d, y, label="data", marker="o", linestyle="none")
    #tp.integer_xticks()
    ##tp.plot(x_fit, y_fit, label="fit")
    #tp.legend()
    #tp.save("max_val_gauss.pdf")