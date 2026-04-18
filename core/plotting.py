"""Plotting utilities for paper experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.utils import ensure_dir


def plot_reliability_diagram(result: dict, metric_name: str, save_path: str | Path | None = None):
    """Plot a reliability diagram with confidence histogram."""
    positions = result["bins"][:-1] + result["bin_size"] / 2.0
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(4, 5),
        dpi=120,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.5]},
    )
    axs[0].bar(positions, result["bin_acc"], width=result["bin_size"], edgecolor="black", label="Accuracy")
    axs[0].bar(
        positions,
        result["gaps"],
        bottom=np.minimum(result["bin_acc"], result["bin_conf"]),
        width=result["bin_size"],
        edgecolor="black",
        hatch="//",
        label="Gap",
    )
    axs[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlim(0, 1)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].text(0.65, 0.08, f"ECE={result['ece']:.2f}")

    total = max(np.sum(result["bin_counts"]), 1)
    axs[1].bar(
        positions,
        result["bin_counts"] / total,
        width=result["bin_size"],
        edgecolor="black",
        color="tab:orange",
    )
    axs[1].axvline(result["avg_acc"], linestyle="--", color="blue", label="Avg. accuracy")
    axs[1].axvline(result["avg_conf"], linestyle="--", color="red", label="Avg. confidence")
    axs[1].set_xlabel(metric_name)
    axs[1].set_ylabel("Fraction")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_curve(x, y, xlabel: str, ylabel: str, save_path: str | Path | None = None):
    """Plot a simple experiment curve."""
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_umap_embedding(embedding, labels, scores=None, save_path: str | Path | None = None):
    """Plot a two-dimensional UMAP embedding."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    color = scores if scores is not None else labels
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=8, cmap="viridis", alpha=0.8)
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
