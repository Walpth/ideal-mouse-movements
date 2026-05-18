from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

DEFAULT_CSV = "strokes_csv/mouse_segmented.csv"
DEFAULT_OUTPUT_DIR = "strokes_segmented"
DEFAULT_SAMPLE_COUNT = 3000
DEFAULT_RANDOM_SEED = 32
TARGET_WIDTH = 20.0
PATHS_CSV = Path("strokes_csv/move_start_end.csv")

SAVE_TRAJECTORIES = True
MAX_TRAJECTORY_FIGURES = 100


def _movement_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    if "event_type" not in df.columns:
        return [df.sort_values("seq")]

    moves = df[df["event_type"] == "move"].copy()
    if moves.empty:
        return []

    if "substroke" in moves.columns:
        return [
            group.sort_values("seq")
            for _, group in moves.groupby("substroke", sort=False)
            if len(group) >= 2
        ]

    moves = moves.sort_values("seq")
    block_id = (moves.index.to_series().diff().fillna(1) != 1).cumsum()
    return [
        group
        for _, group in moves.groupby(block_id, sort=False)
        if len(group) >= 2
    ]


def write_start_end_csv(sampled_strokes: list[tuple[int, pd.DataFrame]], output_csv: Path) -> None:
    """Write start/end coordinates for sampled moves to a CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stroke_id", "start_x", "start_y", "end_x", "end_y"])
        for stroke_id, df in sampled_strokes:
            segments = _movement_segments(df)
            ordered = pd.concat(segments, ignore_index=True) if segments else df.sort_values("seq").reset_index(drop=True)
            if ordered.empty:
                continue
            start_x = float(ordered.iloc[0]["x"])
            start_y = float(ordered.iloc[0]["y"])
            end_x = float(ordered.iloc[-1]["x"])
            end_y = float(ordered.iloc[-1]["y"])
            writer.writerow([stroke_id, f"{start_x:.6f}", f"{start_y:.6f}", f"{end_x:.6f}", f"{end_y:.6f}"])

def compute_speed_from_points(points: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute time and speed arrays from point coordinates.

    If timestamps are provided, speed is computed using timestamp deltas.
    Otherwise, a unit time step is assumed between samples.
    """
    if len(points) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if len(points) == 1:
        t = np.array([0.0], dtype=float)
        speed = np.array([0.0], dtype=float)
        return t, speed

    if timestamps is not None and len(timestamps) == len(points):
        t = np.asarray(timestamps, dtype=float)
        dt = np.diff(t)
        dt = np.where(dt <= 0, np.nan, dt)
        step_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        step_speed = np.divide(step_dist, dt, out=np.zeros_like(step_dist), where=~np.isnan(dt))
        speed = np.concatenate([[0.0], step_speed])
        t = t - t[0]
        return t, speed

    dt = 1.0
    t = np.arange(len(points), dtype=float) * dt
    step_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = np.concatenate([[0.0], step_dist / dt])
    return t, speed


def compute_heading_series(
    points: np.ndarray,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes a smooth relative heading angle θ_rel(t) for PCA training.

    θ_rel(t) = θ(t) − chord_heading

    where θ(t) = atan2(vy, vx) unwrapped to remove 2π discontinuities, and
    chord_heading is the straight-line angle from first to last point.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        return np.zeros(len(points), dtype=float)

    window = min(20, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window < 5:
        window = 5

    smoothed = savgol_filter(points, window_length=window, polyorder=3, axis=0, mode='mirror')

    t = np.asarray(times, dtype=float).copy() if times is not None else np.arange(len(points), dtype=float)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-9

    vx = np.gradient(smoothed[:, 0], t, edge_order=2)
    vy = np.gradient(smoothed[:, 1], t, edge_order=2)

    vx = savgol_filter(vx, window, 2, mode='mirror')
    vy = savgol_filter(vy, window, 2, mode='mirror')

    theta = np.unwrap(np.arctan2(vy, vx))

    chord_heading = math.atan2(
        float(points[-1, 1] - points[0, 1]),
        float(points[-1, 0] - points[0, 0]),
    )
    theta_rel = theta - chord_heading
    theta_rel -= theta_rel[0]

    return theta_rel


def compute_segmented_heading_series(df: pd.DataFrame, t: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in _movement_segments(df):
        idx = segment.index.to_numpy(dtype=int)
        if len(idx) < 5 or idx.max(initial=-1) >= len(t):
            continue
        points = segment[["x", "y"]].to_numpy(dtype=float)
        seg_t = t[idx]
        heading = compute_heading_series(points, seg_t)
        series.append((seg_t, heading))
    return series


def _normalize_timestamps_to_seconds(series: pd.Series) -> pd.Series:
    """
    Detect timestamp unit by median value and return a new series in
    relative seconds (offset so the global first event = 0).

    Thresholds on median:
        > 1e15: nanoseconds
        > 1e12: microseconds
        > 1e9: milliseconds
        else: seconds
    """
    med = series.median()
    if med > 1e15:
        scale = 1e9
    elif med > 1e12:
        scale = 1e6
    elif med > 1e9:
        scale = 1e3
    else:
        scale = 1.0

    ts = series / scale

    # Convert epoch-based seconds to a relative offset.
    if ts.median() > 1e8:
        ts = ts - ts.iloc[0]

    if scale != 1.0:
        print(f"  [load_strokes] timestamp unit detected as x{scale:.0e}; normalized to relative seconds.")

    return ts


def load_strokes(csv_path: Path) -> dict[int, pd.DataFrame]:
    """Load the CSV and return a mapping from stroke_id to sorted DataFrame."""
    df = pd.read_csv(csv_path)

    required = {"stroke_id", "seq", "timestamp", "x", "y", "speed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Normalise timestamps once so all downstream code works in seconds.
    df["timestamp"] = _normalize_timestamps_to_seconds(df["timestamp"])

    strokes: dict[int, pd.DataFrame] = {}
    for stroke_id, group in df.groupby("stroke_id", sort=False):
        g = group.sort_values("seq").reset_index(drop=True)
        strokes[int(stroke_id)] = g
    return strokes


def choose_random_strokes(strokes: dict[int, pd.DataFrame], sample_count: int, seed: int) -> list[tuple[int, pd.DataFrame]]:
    ids = np.array(list(strokes.keys()), dtype=int)
    if len(ids) == 0:
        return []
    return [(int(stroke_id), strokes[int(stroke_id)]) for stroke_id in ids]


def format_metrics_text(
    stroke_id: int,
    points: np.ndarray,
    speed: np.ndarray,
    timestamps: np.ndarray,
    df: pd.DataFrame,
) -> str:
    start = points[0]
    end = points[-1]
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    total_distance = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0
    displacement = float(np.linalg.norm(end - start))
    avg_speed = float(np.mean(speed)) if len(speed) else 0.0
    median_speed = float(np.median(speed)) if len(speed) else 0.0
    max_speed = float(np.max(speed)) if len(speed) else 0.0
    min_speed = float(np.min(speed)) if len(speed) else 0.0

    target_x = end[0]
    target_y = end[1]
    if {"button_held", "event_type"}.issubset(df.columns):
        event_type = str(df["event_type"].iloc[0])
    else:
        event_type = "unknown"

    return (
        f"stroke_id: {stroke_id}    "
        f"points: {len(points)}    "
        f"duration: {duration:.3f} s    "
        f"path distance: {total_distance:.1f} px    "
        f"displacement: {displacement:.1f} px\n"
        f"start: ({start[0]:.1f}, {start[1]:.1f})    "
        f"end: ({end[0]:.1f}, {end[1]:.1f})    "
        f"avg speed: {avg_speed:.1f} px/s    "
        f"median speed: {median_speed:.1f} px/s    "
        f"max speed: {max_speed:.1f} px/s    "
        f"min speed: {min_speed:.1f} px/s    "
        f"event type: {event_type}"
    )


def save_stroke_figure(
    stroke_id: int,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    df = df.sort_values("seq").reset_index(drop=True)
    points = df[["x", "y"]].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=float) if "timestamp" in df.columns else None

    if "speed" in df.columns:
        speed = df["speed"].to_numpy(dtype=float)
        if len(speed) != len(points):
            speed = speed[: len(points)]
        t = (timestamps - timestamps[0]) if timestamps is not None and len(timestamps) else np.arange(len(points), dtype=float)
    else:
        t, speed = compute_speed_from_points(points, timestamps)

    if timestamps is None or len(timestamps) != len(points):
        timestamps = np.arange(len(points), dtype=float)
        
    heading_segments = compute_segmented_heading_series(df, t)
    substroke_boundary_idx = np.array([], dtype=int)
    if "substroke" in df.columns:
        substrokes = df["substroke"].to_numpy()
        substroke_boundary_idx = np.flatnonzero(substrokes != np.roll(substrokes, 1))
        substroke_boundary_idx = substroke_boundary_idx[substroke_boundary_idx > 0]
    stop_spans: list[tuple[float, float]] = []
    if "event_type" in df.columns and "dt" in df.columns:
        stop_rows = df.index[df["event_type"] == "stop"].to_numpy(dtype=int)
        dt_values = df["dt"].to_numpy(dtype=float)
        for idx in stop_rows:
            if idx < len(t):
                stop_spans.append((float(t[idx] - dt_values[idx]), float(t[idx])))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        3, 
        2, 
        height_ratios=[2.0, 2.0, 1.2], 
        width_ratios=[3.2, 2.0], 
        hspace=0.35, 
        wspace=0.25
    )

    ax_path = fig.add_subplot(gs[0:2, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_angular = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    sc = ax_path.scatter(
        points[:, 0],
        points[:, 1],
        c=speed[: len(points)],
        cmap="viridis",
        s=18,
        linewidths=0.0,
        alpha=0.95,
        zorder=3,
    )

    ax_path.plot(points[:, 0], points[:, 1], color="0.7", linewidth=0.8, alpha=0.35, zorder=2)
    ax_path.plot(points[0, 0], points[0, 1], "go", markersize=10, label="Start", zorder=5)
    ax_path.plot(points[-1, 0], points[-1, 1], "r^", markersize=10, label="End", zorder=5)
    if len(substroke_boundary_idx):
        ax_path.plot(
            points[substroke_boundary_idx, 0],
            points[substroke_boundary_idx, 1],
            "rx",
            markersize=9,
            markeredgewidth=2,
            label="Substroke boundary",
            zorder=6,
        )

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_pad = (x_max - x_min) * 0.15 + 10
    y_pad = (y_max - y_min) * 0.15 + 10
    ax_path.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_path.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.25)
    ax_path.set_xlabel("X (pixels)")
    ax_path.set_ylabel("Y (pixels)")
    ax_path.tick_params(labelsize=8)
    ax_path.set_title(f"Stroke {stroke_id} | speed-colored path", fontsize=10, fontweight="bold")
    ax_path.legend(fontsize=8, loc="best")

    cbar = fig.colorbar(sc, ax=ax_path, fraction=0.046, pad=0.04)
    cbar.set_label("Pointer speed (px/s)")

    ax_speed.plot(t[:len(speed)], speed, linewidth=1.6)
    for start_t, end_t in stop_spans:
        ax_speed.axvspan(start_t, end_t, color="green", alpha=0.25)
        ax_speed.axvline(start_t, color="red", linewidth=1.0, alpha=0.85)
        ax_speed.axvline(end_t, color="red", linewidth=1.0, alpha=0.85)
    for idx in substroke_boundary_idx:
        if idx < len(t):
            ax_speed.axvline(t[idx], color="red", linewidth=1.0, alpha=0.85)
    ax_speed.set_title("Speed profile", fontsize=10, fontweight="bold")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (px/s)")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.tick_params(labelsize=8)
    
    if heading_segments:
        for seg_t, heading in heading_segments:
            ax_angular.plot(seg_t[:len(heading)], heading, linewidth=1.6, color="#16a34a")
    else:
        ax_angular.text(0.5, 0.5, "Not enough move data", ha="center", va="center", transform=ax_angular.transAxes)
    for start_t, end_t in stop_spans:
        ax_angular.axvspan(start_t, end_t, color="green", alpha=0.15)
    ax_angular.set_title("Heading angle profile", fontsize=10, fontweight="bold")
    ax_angular.set_xlabel("Time (s)")
    ax_angular.set_ylabel("Heading angle (rad)")
    ax_angular.grid(True, alpha=0.25)
    ax_angular.tick_params(labelsize=8)

    metrics_text = format_metrics_text(
        stroke_id=stroke_id,
        points=points,
        speed=speed,
        timestamps=timestamps,
        df=df,
    )
    ax_text.text(
        0.01,
        0.65,
        metrics_text,
        ha="left",
        va="center",
        fontsize=9,
        family="monospace",
        wrap=True,
    )

    fig.suptitle("Random sample mouse movement stroke", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _path_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _segment_speeds(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([], dtype=float)
    if len(times) != len(points):
        return np.array([], dtype=float)
    dt = np.diff(times)
    dt = np.where(dt <= 0, 1e-9, dt)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return dists / dt


def _resample_profile(progress: list[float], values: list[float], sample_count: int = 101) -> list[float]:
    if not progress or not values:
        return []
    x = np.array(progress, dtype=float)
    y = np.array(values, dtype=float)
    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    if x[-1] < 1.0:
        x = np.append(x, 1.0)
        y = np.append(y, y[-1])
    grid = np.linspace(0.0, 1.0, sample_count)
    return np.interp(grid, x, y).tolist()


def _analyze_sampled_strokes(sampled_strokes: list[tuple[int, pd.DataFrame]], output_dir: Path) -> dict[str, object]:
    per_move_features: list[dict[str, float]] = []
    abs_profiles: list[list[float]] = []
    norm_profiles: list[list[float]] = []
    curv_profiles: list[list[float]] = []

    for stroke_id, df in sampled_strokes:
        df = df.sort_values("seq").reset_index(drop=True)
        for segment in _movement_segments(df):
            segment = segment.sort_values("seq")
            points = segment[["x", "y"]].to_numpy(dtype=float)
            timestamps = segment["timestamp"].to_numpy(dtype=float) if "timestamp" in segment.columns else np.arange(len(points), dtype=float)
            t = timestamps - timestamps[0] if len(timestamps) > 1 else np.arange(len(points), dtype=float)
            if len(points) < 2 or len(timestamps) != len(points):
                continue

            speeds = _segment_speeds(points, timestamps)
            if len(speeds) < 2:
                continue

            duration_sec = float(timestamps[-1] - timestamps[0])
            if duration_sec <= 0:
                continue

            path_distance_px = _path_distance(points)
            if path_distance_px <= 0:
                continue

            start = points[0]
            end = points[-1]
            net_displacement = float(np.linalg.norm(end - start))
            straightness = net_displacement / path_distance_px if path_distance_px > 0 else 0.0

            per_move_features.append(
                {
                    "stroke_id": float(stroke_id),
                    "duration_sec": duration_sec,
                    "path_distance_px": path_distance_px,
                    "straightness_ratio": straightness,
                }
            )

            progress = ((timestamps - timestamps[0]) / duration_sec).clip(0.0, 1.0).tolist()
            peak_speed = float(np.max(speeds)) if len(speeds) else 0.0
            normalized_speed = (speeds / peak_speed).tolist() if peak_speed > 0 else [0.0 for _ in speeds]

            angular_velocity_series = compute_heading_series(points, t)

            abs_profiles.append(_resample_profile(progress[1:], speeds.tolist()))
            norm_profiles.append(_resample_profile(progress[1:], normalized_speed))
            curv_profiles.append(_resample_profile(progress[1:], angular_velocity_series[1:].tolist()))

    try:
        with (output_dir / "moves-human.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["distance", "duration"])
            for mv in per_move_features:
                writer.writerow([f"{mv['path_distance_px']:.6f}", f"{mv['duration_sec']:.6f}"])
    except Exception:
        pass

    if per_move_features:
        straightness_values = [item["straightness_ratio"] for item in per_move_features]
        anti_macro = {
            "move_count": len(per_move_features),
            "median_straightness_ratio": float(statistics.median(straightness_values)),
            "very_straight_move_ratio": float(sum(1 for value in straightness_values if value >= 0.985) / len(straightness_values)),
        }
    else:
        anti_macro = {"move_count": 0, "median_straightness_ratio": None, "very_straight_move_ratio": None}

    if norm_profiles:
        norm_array = np.array(norm_profiles, dtype=float)
        abs_array = np.array(abs_profiles, dtype=float)
        curv_array = np.array(curv_profiles, dtype=float)
        
        profile = {
            "progress_0_1": np.linspace(0.0, 1.0, norm_array.shape[1]).tolist(),
            "mean_speed_px_s": np.mean(abs_array, axis=0).tolist(),
            "mean_speed_normalized": np.mean(norm_array, axis=0).tolist(),
            "p10_speed_normalized": np.quantile(norm_array, 0.1, axis=0).tolist(),
            "p90_speed_normalized": np.quantile(norm_array, 0.9, axis=0).tolist(),
            "mean_angular_velocity": np.mean(curv_array, axis=0).tolist(),
            "p10_angular_velocity": np.quantile(curv_array, 0.1, axis=0).tolist(),
            "p90_angular_velocity": np.quantile(curv_array, 0.9, axis=0).tolist(),
            "moves_used": int(norm_array.shape[0]),
        }
    else:
        profile = {
            "progress_0_1": [], "mean_speed_px_s": [], "mean_speed_normalized": [], 
            "p10_speed_normalized": [], "p90_speed_normalized": [], 
            "mean_angular_velocity": [], "p10_angular_velocity": [], "p90_angular_velocity": [],
            "moves_used": 0
        }

    return {"anti_macro": anti_macro, "average_move_profile": profile}


def _render_analysis_figure(analysis: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)
    fig.patch.set_facecolor("#f9fafb")

    anti_macro = analysis["anti_macro"]
    profile = analysis["average_move_profile"]

    ax_anti, ax_curve, ax_curv = axes

    median_s = anti_macro.get("median_straightness_ratio")
    very_s = anti_macro.get("very_straight_move_ratio")
    labels = ["Median straightness", "Very-straight ratio"]
    vals = [100.0 * (median_s if median_s is not None else 0.0), 100.0 * (very_s if very_s is not None else 0.0)]
    colors = ["#7c3aed", "#a78bfa"]
    ax_anti.barh(labels, vals, color=colors, alpha=0.9)
    ax_anti.set_xlim(0, 100)
    ax_anti.set_xlabel("percent (0-100)")
    ax_anti.set_title("Straightness Summary")
    ax_anti.grid(axis="x", alpha=0.2)

    progress = profile.get("progress_0_1", [])
    mean_norm = profile.get("mean_speed_normalized", [])
    p10_norm = profile.get("p10_speed_normalized", [])
    p90_norm = profile.get("p90_speed_normalized", [])
    mean_abs = profile.get("mean_speed_px_s", [])
    mean_angular = profile.get("mean_angular_velocity", [])
    p10_angular = profile.get("p10_angular_velocity", [])
    p90_angular = profile.get("p90_angular_velocity", [])

    if progress and mean_norm:
        ax_curve.plot(progress, mean_norm, color="#7c3aed", linewidth=2.0, label="mean normalized speed")
        ax_curve.fill_between(progress, p10_norm, p90_norm, color="#a78bfa", alpha=0.25, label="p10-p90 band")
        ax_curve.set_xlim(0.0, 1.0)
        ax_curve.set_ylim(0.0, max(1.05, max(p90_norm) * 1.05))
        if mean_abs:
            ax_curve2 = ax_curve.twinx()
            ax_curve2.plot(progress, mean_abs, color="#ea580c", linewidth=1.1, alpha=0.7, label="mean absolute speed")
            ax_curve2.set_ylabel("mean speed (px/s)")
    else:
        ax_curve.text(0.5, 0.5, "Not enough move data", ha="center", va="center", transform=ax_curve.transAxes)

    ax_curve.set_title("Average Speed Shape Within One Move")
    ax_curve.set_xlabel("normalized move progress (0=start, 1=end)")
    ax_curve.set_ylabel("normalized speed")
    ax_curve.grid(alpha=0.2)
    
    if progress and mean_angular:
        ax_curv.plot(progress, mean_angular, color="#16a34a", linewidth=2.0, label="mean heading angle")
        ax_curv.fill_between(progress, p10_angular, p90_angular, color="#4ade80", alpha=0.25, label="p10-p90 band")
        ax_curv.set_xlim(0.0, 1.0)
    else:
        ax_curv.text(0.5, 0.5, "Not enough move data", ha="center", va="center", transform=ax_curv.transAxes)

    ax_curv.set_title("Average Heading Angle Shape Within One Move")
    ax_curv.set_xlabel("normalized move progress (0=start, 1=end)")
    ax_curv.set_ylabel("heading angle (rad)")
    ax_curv.legend()
    ax_curv.grid(alpha=0.2)

    fig.suptitle(f"CSV path analysis | moves analyzed={anti_macro.get('move_count', 0)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_fitts_plot(csv_path: Path, output_path: Path, W: float = TARGET_WIDTH) -> None:
    try:
        data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
        if getattr(data, "size", 0) == 0:
            return
        x = np.asarray(data["distance"], dtype=float)
        y = np.asarray(data["duration"], dtype=float)
    except Exception:
        try:
            arr = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
            if arr.size == 0:
                return
            if arr.ndim == 1 and arr.shape[0] >= 2:
                x = np.array([arr[0]], dtype=float)
                y = np.array([arr[1]], dtype=float)
            else:
                x = arr[:, 0]
                y = arr[:, 1]
        except Exception:
            return

    mask = (x >= 5) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return

    fitts_x = np.log2(1 + x / W)
    b_fitts, a_fitts = np.polyfit(fitts_x, y, 1)

    x_smooth = np.linspace(x.min(), x.max(), 300)
    fitts_x_smooth = np.log2(1 + x_smooth / W)
    y_fitts = a_fitts + b_fitts * fitts_x_smooth
    y_fit = a_fitts + b_fitts * fitts_x
    sigma = float(np.std(y - y_fit))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.scatter(x, y, label="Data", alpha=0.8)
    ax.plot(x_smooth, y_fitts, label="Fitts-like fit", linewidth=2)
    ax.fill_between(x_smooth, y_fitts - sigma, y_fitts + sigma, alpha=0.2, label="±1σ")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2)
    ax.set_xlabel("Distance (px)")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Fitts-like Fit (Linear Scale)")
    ax.legend()
    ax.grid(True)
    model_text = f"Model: duration ≈ {a_fitts:.3f} + {b_fitts:.3f} * log2(1 + D/{W})\nSigma: {sigma:.3f}"
    ax.text(0.98, 0.02, model_text, transform=ax.transAxes, fontsize=9, ha="right", va="bottom", family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Model: duration ≈ {a_fitts:.3f} + {b_fitts:.3f} * log2(1 + D/{W})")
    print(f"Sigma: {sigma:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random mouse movement samples from a CSV file.")
    parser.add_argument("csv_file", nargs="?", default=DEFAULT_CSV, help="Input CSV file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output PNG files")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of random strokes to sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strokes = load_strokes(csv_path)
    chosen = choose_random_strokes(strokes, args.count, args.seed)

    if not chosen:
        raise RuntimeError(f"No strokes found in CSV: {csv_path}")

    print(f"Loaded {len(strokes)} strokes from {csv_path}")
    print(f"Sampling {len(chosen)} stroke(s) with seed={args.seed}")

    write_start_end_csv(chosen, PATHS_CSV)
    print(f"Saved {PATHS_CSV}")

    analysis = _analyze_sampled_strokes(chosen, output_dir)
    analysis_path = output_dir / "generated_paths_analysis.png"
    _render_analysis_figure(analysis, analysis_path)
    print(f"Saved {analysis_path}")

    fitts_csv = output_dir / "moves-human.csv"
    fitts_path = output_dir / "distance_duration_fitts.png"
    _render_fitts_plot(fitts_csv, fitts_path)

    if SAVE_TRAJECTORIES:
        subdir = output_dir / "trajectories"
        subdir.mkdir(parents=True, exist_ok=True)
        for stroke_id, df in chosen[:MAX_TRAJECTORY_FIGURES]:
            out_path = subdir / f"stroke_{stroke_id:06d}.png"
            save_stroke_figure(stroke_id, df, out_path)
            print(f"Saved {out_path}")    

    print(f"Done. Images saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
