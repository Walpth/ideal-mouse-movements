#!/usr/bin/env python3
"""Measure PCA variance for the current training architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from main_pipeline.train import (
    _decompose_hierarchical,
    _iter_move_substrokes,
    _relative_seconds,
    _resample,
    compute_theta_arclength,
)


LEVEL_SIZES = [9, 17, 33, 65, 129]
LEVEL_COMPONENTS = [6, 8, 16, 28, 16]
PHASE_GRID = np.linspace(0.0, 1.0, 129)


def collect_training_samples(csv_path: Path) -> list[dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    samples: list[dict[str, np.ndarray]] = []

    for stroke in _iter_move_substrokes(df):
        if len(stroke) < 10:
            continue

        t = stroke["timestamp"].to_numpy()
        speed = stroke["speed"].to_numpy(dtype=float)
        points = stroke[["x", "y"]].to_numpy(dtype=float)

        t_seconds = _relative_seconds(t)
        duration = max(float(t_seconds[-1] - t_seconds[0]), 1e-3)
        distance = max(float(np.linalg.norm(points[-1] - points[0])), 1.0)

        normalized_speed = speed / (distance / duration)
        speed_profile = np.arcsinh(normalized_speed)

        s_norm, theta = compute_theta_arclength(points)
        samples.append({
            "v_asinh": _resample(s_norm, speed_profile, PHASE_GRID),
            "theta": _resample(s_norm, theta, PHASE_GRID),
        })

    if not samples:
        raise RuntimeError(f"No movement substrokes found in {csv_path}")

    return samples


def analyze_level_variance(samples: list[dict[str, np.ndarray]], threshold: float) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for level, (level_size, architecture_k) in enumerate(zip(LEVEL_SIZES, LEVEL_COMPONENTS)):
        speed_matrix = np.stack([
            _decompose_hierarchical(sample["v_asinh"], LEVEL_SIZES)[level]
            for sample in samples
        ])
        theta_matrix = np.stack([
            _decompose_hierarchical(sample["theta"], LEVEL_SIZES)[level]
            for sample in samples
        ])

        speed_std = max(float(np.std(speed_matrix)), 1e-4)
        theta_std = max(float(np.std(theta_matrix)), 1e-4)
        joint = np.concatenate([speed_matrix / speed_std, theta_matrix / theta_std], axis=1)

        max_components = min(joint.shape[0] - 1, joint.shape[1])
        if max_components < 1:
            continue

        pca = PCA(n_components=max_components).fit(joint)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        effective_k = min(architecture_k, max_components)
        target_k = int(np.searchsorted(cumulative, threshold) + 1)
        target_k = min(target_k, max_components)

        results.append({
            "level": level,
            "level_size": level_size,
            "architecture_k": architecture_k,
            "effective_k": effective_k,
            "max_components": max_components,
            "explained_at_architecture_k": float(cumulative[effective_k - 1]),
            "target_k": target_k,
            "threshold": threshold,
            "cumulative": cumulative,
        })

    return results


def print_summary(results: list[dict[str, object]]) -> None:
    print("PCA variance for main_pipeline/train.py architecture")
    rows = [
        (
            str(result["level_size"]),
            str(result["effective_k"]),
            f"{result['explained_at_architecture_k']:.3f}",
            str(result["target_k"]),
        )
        for result in results
    ]
    headers = ("level size", "train k", "explained at train k", "k for target")
    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]

    print(" | ".join(f"{header:>{width}}" for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(f"{value:>{width}}" for value, width in zip(row, widths)))


def plot_summary(results: list[dict[str, object]], save_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for result in results:
        cumulative = np.asarray(result["cumulative"], dtype=float)
        ks = np.arange(1, len(cumulative) + 1)
        label = f"level {result['level_size']} (k={result['effective_k']})"
        ax.plot(ks, cumulative, marker="o", linewidth=1.5, markersize=3, label=label)
        ax.axvline(int(result["effective_k"]), color="0.75", linewidth=0.8)

    threshold = float(results[0]["threshold"]) if results else 0.95
    ax.axhline(threshold, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA variance by main_pipeline/train.py level")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PCA variance for main_pipeline/train.py's current architecture.")
    parser.add_argument("csv_path", nargs="?", default="strokes_csv/mouse_segmented.csv")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output", default="explained_variance.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    samples = collect_training_samples(Path(args.csv_path))
    results = analyze_level_variance(samples, args.threshold)
    print_summary(results)
    plot_summary(results, Path(args.output), args.show)


if __name__ == "__main__":
    main()
