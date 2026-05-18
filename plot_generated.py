from __future__ import annotations
import statistics
import csv
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from main_pipeline.builder import build_trajectory
from scipy.signal import savgol_filter

OUTPUT_DIR = Path("strokes_generated")
ANALYSIS_OUTPUT = OUTPUT_DIR / "generated_paths_analysis.png"
MOVES_CSV = OUTPUT_DIR / "moves-human.csv"
Fitts_OUTPUT = OUTPUT_DIR / "distance_duration_fitts.png"
SAMPLE_RATE = 1000
TARGET_WIDTH = 1.0
PATHS_CSV = Path("strokes_csv/move_start_end.csv")

SAVE_TRAJECTORIES = True
MAX_TRAJECTORY_FIGURES = 100


def load_start_end_pairs(csv_path: Path) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """Load start/end coordinates from CSV rows, including stroke_id."""
    required_columns = {"start_x", "start_y", "end_x", "end_y"}
    pairs: list[tuple[str, tuple[float, float], tuple[float, float]]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns.difference(fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader):
            try:
                stroke_id = str(row.get("stroke_id", f"unknown_stroke_{i}"))
                start = (float(row["start_x"]), float(row["start_y"]))
                end = (float(row["end_x"]), float(row["end_y"]))
            except (TypeError, ValueError):
                continue
            pairs.append((stroke_id, start, end))

    return pairs

def compute_speed(points: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return np.array([0.0]), np.array([0.0])
    dt = 1.0 / sample_rate
    deltas = np.diff(points, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    speed = np.concatenate([[0.0], step_dist / dt])
    t = np.arange(len(points), dtype=float) * dt
    return t, speed

def get_speed_series(traj: dict, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(traj["points"], dtype=float)
    times_raw = traj.get("times")
    if times_raw is not None:
        t = np.asarray(times_raw, dtype=float)
        if len(t) == len(points) and len(t) > 1:
            dt = np.diff(t)
            dt = np.where(dt <= 0, 1e-9, dt)
            dists = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
            speed_px_s = np.concatenate([[0.0], dists / dt])
            return t, speed_px_s
    return compute_speed(points, sample_rate)

def _strictly_increasing_times(times: np.ndarray, n: int) -> np.ndarray:
    t = np.asarray(times, dtype=float).copy() if times is not None else np.arange(n, dtype=float)
    if len(t) != n:
        t = np.arange(n, dtype=float)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-9
    return t

def _resample_optional_series(series, n: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        return np.zeros(n, dtype=float)
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(values)), values)

def compute_angular_velocity_series(
    points: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Computes a smooth relative heading angle θ_rel(t) for PCA training.

    θ_rel(t) = θ(t) − chord_heading

    where θ(t) = atan2(vy, vx) unwrapped to remove 2π discontinuities, and
    chord_heading is the straight-line angle from first to last point.

    Properties:
        - no compression needed: θ is naturally bounded by total heading excursion
        - sharp turns appear as clean steps/ramps instead of biphasic spike pairs
        - fully invertible at generation time: θ(t) = chord_heading + θ_rel(t),
          then ω(t) = dθ/dt for the existing integrator (or integrate θ directly)
        - sign-preserving: left turns positive, right turns negative (atan2 convention)
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        return np.zeros(len(points), dtype=float)

    window = min(20, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window < 5:
        window = 5

    smoothed = savgol_filter(points, window_length=window, polyorder=3, axis=0, mode='mirror')

    t = _strictly_increasing_times(times, len(points))

    vx = np.gradient(smoothed[:, 0], t, edge_order=2)
    vy = np.gradient(smoothed[:, 1], t, edge_order=2)

    vx = savgol_filter(vx, window, 2, mode='mirror')
    vy = savgol_filter(vy, window, 2, mode='mirror')

    # atan2 has 2π jumps at heading reversals; unwrap makes the series continuous
    # so PCA sees smooth ramps instead of sawtooth discontinuities.
    theta = np.unwrap(np.arctan2(vy, vx))

    # At generation time the chord heading is supplied by the task geometry, so
    # subtracting it here makes the feature direction-agnostic across trajectories.
    chord_heading = math.atan2(
        float(points[-1, 1] - points[0, 1]),
        float(points[-1, 0] - points[0, 0]),
    )
    theta_rel = theta - chord_heading

    # Re-centre so the series starts near zero (removes any constant offset from
    # the unwrap anchor combined with the chord subtraction).
    theta_rel -= theta_rel[0]

    return theta_rel


def save_trajectory_figure(
    traj: dict,
    start: np.ndarray,
    output_path: Path,
    idx: int,
    stroke_id: str,
    sample_rate: float,
) -> None:
    points = np.asarray(traj["points"], dtype=float)
    start = np.asarray(start, dtype=float)
    clean_path = np.asarray(traj["path"], dtype=float)
    target = np.asarray(traj["nominal_end"], dtype=float)
    actual_end = np.asarray(traj["biased_end"], dtype=float)

    t, speed = get_speed_series(traj, sample_rate)
    angular_velocity = compute_angular_velocity_series(points, t)
    original_angular_velocity = traj.get("theta_series_orig", [0.0] * len(points))
    original_angular_velocity = _resample_optional_series(original_angular_velocity, len(points))

    dev_vec = actual_end - target
    dev_dist = float(np.linalg.norm(dev_vec))
    start_to_target = float(np.linalg.norm(target - start))
    target_to_end = float(np.linalg.norm(actual_end - target))
    duration = float(traj.get("duration", t[-1] if len(t) else 0.0))
    distance = float(traj.get("dist", start_to_target))

    avg_speed = float(np.mean(speed)) if len(speed) else 0.0
    median_speed = float(np.median(speed)) if len(speed) else 0.0
    max_speed = float(np.max(speed)) if len(speed) else 0.0

    plan = traj.get("plan", {})
    stroke_type = plan.get("stroke_type", "unknown")
    velocity_intent = plan.get("velocity_intent", "unknown")
    needs_correction = plan.get("needs_correction", "unknown")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 2.0, 1.2], width_ratios=[3.2, 2.0], hspace=0.35, wspace=0.25)

    ax_path = fig.add_subplot(gs[0:2, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_angular = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    sc = ax_path.scatter(points[:, 0], points[:, 1], c=speed[: len(points)], cmap="viridis", s=18, linewidths=0.0, alpha=0.95, zorder=3)

    ax_path.plot(clean_path[:, 0], clean_path[:, 1], linestyle="--", color="0.6", linewidth=1.0, alpha=0.45, zorder=2, label="Clean path")

    ax_path.plot(start[0], start[1], "go", markersize=10, label="Start", zorder=5)
    ax_path.plot(target[0], target[1], "r*", markersize=16, markeredgecolor="darkred", markeredgewidth=1.2, label="Nominal target", zorder=6)
    ax_path.plot(actual_end[0], actual_end[1], "c^", markersize=10, markeredgecolor="darkblue", markeredgewidth=1.0, label="Actual endpoint", zorder=6)

    if dev_dist > 2:
        ax_path.arrow(target[0], target[1], dev_vec[0], dev_vec[1], head_width=5, head_length=4, fc="orange", ec="orange", alpha=0.65, zorder=4, length_includes_head=True)

    all_points = np.vstack([points, clean_path, [target], [actual_end], [start]])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = (x_max - x_min) * 0.15 + 20
    y_pad = (y_max - y_min) * 0.15 + 20

    ax_path.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_path.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.25)
    ax_path.set_xlabel("X (pixels)")
    ax_path.set_ylabel("Y (pixels)")
    ax_path.tick_params(labelsize=8)
    ax_path.set_title(f"Path {idx + 1} | {stroke_id} | {stroke_type} | speed-colored dots", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax_path, fraction=0.046, pad=0.04)
    cbar.set_label("Pointer speed (px/s)")
    ax_path.legend(fontsize=8, loc="best")

    ax_speed.plot(t[:len(speed)], speed, linewidth=1.6)
    ax_speed.set_title("Speed profile", fontsize=10, fontweight="bold")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (px/s)")
    ax_speed.set_ylim(0, max(speed) * 1.1 if len(speed) > 0 else 1.0) # PATCH: Fix negative padding
    ax_speed.grid(True, alpha=0.25)
    ax_speed.tick_params(labelsize=8)

    ax_angular.plot(t[:len(angular_velocity)], angular_velocity, linewidth=1.6, color="#16a34a")
    ax_angular.plot(t[:len(original_angular_velocity)], original_angular_velocity, linewidth=1.6, color="#EC1A1A", alpha=0.7)
    ax_angular.set_title("Heading angle profile", fontsize=10, fontweight="bold")
    ax_angular.set_xlabel("Time (s)")
    ax_angular.set_ylabel("Heading angle (rad)")
    ax_angular.grid(True, alpha=0.25)
    ax_angular.tick_params(labelsize=8)

    metrics_text = (
        f"Stroke ID: {stroke_id}    "
        f"Start to target distance: {start_to_target:.1f} px    "
        f"Target to actual end distance: {target_to_end:.1f} px    "
        f"Deviation vector: ({dev_vec[0]:.1f}, {dev_vec[1]:.1f}) px    "
        f"Duration: {duration:.3f} s\n"
        f"Mean speed: {avg_speed:.1f} px/s    "
        f"Median speed: {median_speed:.1f} px/s    "
        f"Max speed: {max_speed:.1f} px/s    "
        f"Stroke type: {stroke_type}    "
        f"Velocity intent: {velocity_intent}    "
        f"Needs correction: {needs_correction}    "
        f"Model distance: {distance:.1f} px"
    )
    ax_text.text(0.01, 0.65, metrics_text, ha="left", va="center", fontsize=9, family="monospace", wrap=True)

    fig.suptitle("Sample mouse movement path", fontsize=14, fontweight="bold", y=0.98)
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


def _analyze_generated_trajectories(trajectories: list[dict]) -> dict[str, object]:
    per_move_features: list[dict[str, float]] = []
    abs_profiles: list[list[float]] = []
    norm_profiles: list[list[float]] = []
    curv_profiles: list[list[float]] = []
    orig_curv_profiles: list[list[float]] = []

    for traj in trajectories:
        points = np.asarray(traj["points"], dtype=float)
        times = np.asarray(traj.get("times", []), dtype=float)
        if len(times) != len(points):
            continue
        finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(times)
        points = points[finite_mask]
        times = times[finite_mask]
        if len(points) < 2 or len(times) != len(points):
            continue

        speeds = _segment_speeds(points, times)
        # speeds = traj.get("speed_series_orig", [0.0] * len(speeds_orig))
        # speeds = np.interp(np.linspace(0.0, 1.0, len(speeds_orig)), np.linspace(0.0, 1.0, len(speeds)), speeds)
        if len(speeds) < 2 or not np.isfinite(speeds).all():
            continue

        duration_sec = float(times[-1] - times[0])
        if duration_sec <= 0:
            continue

        path_distance_px = _path_distance(points)
        if path_distance_px <= 0:
            continue

        start = points[0]
        end = points[-1]
        net_displacement = float(np.linalg.norm(end - start))
        straightness = net_displacement / path_distance_px if path_distance_px > 0 else 0.0

        per_move_features.append({
            "duration_sec": duration_sec,
            "path_distance_px": path_distance_px,
            "straightness_ratio": straightness,
        })

        progress = ((times - times[0]) / duration_sec).clip(0.0, 1.0).tolist()
        peak_speed = float(np.max(speeds)) if len(speeds) else 0.0
        normalized_speed = (speeds / peak_speed).tolist() if peak_speed > 0 else [0.0 for _ in speeds]
        
        # Calculate local heading angle for profiles
        angular_velocity_series = compute_angular_velocity_series(points, times)
        if not np.isfinite(angular_velocity_series).all():
            continue

        angular_velocity_series_orig = _resample_optional_series(traj.get("theta_series_orig", []), len(points))

        abs_profiles.append(_resample_profile(progress[1:], speeds.tolist()))
        norm_profiles.append(_resample_profile(progress[1:], normalized_speed))
        curv_profiles.append(_resample_profile(progress[1:], angular_velocity_series[1:].tolist()))
        orig_curv_profiles.append(_resample_profile(progress[1:], angular_velocity_series_orig[1:].tolist()))

    try:
        with MOVES_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["distance", "duration"])
            for mv in per_move_features:
                writer.writerow([f"{mv['path_distance_px']:.6f}", f"{mv['duration_sec']:.6f}"])
    except Exception:
        pass

    if per_move_features:
        straightness_values = [item["straightness_ratio"] for item in per_move_features]
        very_straight_ratio = sum(1 for value in straightness_values if value >= 0.985) / len(straightness_values)
        anti_macro = {
            "move_count": len(per_move_features),
            "median_straightness_ratio": float(statistics.median(straightness_values)),
            "very_straight_move_ratio": float(very_straight_ratio),
        }
    else:
        anti_macro = {"move_count": 0, "median_straightness_ratio": None, "very_straight_move_ratio": None}

    if norm_profiles:
        norm_array = np.array(norm_profiles, dtype=float)
        abs_array = np.array(abs_profiles, dtype=float)
        curv_array = np.array(curv_profiles, dtype=float)
        orig_curv_array = np.array(orig_curv_profiles, dtype=float)
        profile = {
            "progress_0_1": np.linspace(0.0, 1.0, norm_array.shape[1]).tolist(),
            "mean_speed_px_s": np.mean(abs_array, axis=0).tolist(),
            "mean_speed_normalized": np.mean(norm_array, axis=0).tolist(),
            "p10_speed_normalized": np.quantile(norm_array, 0.1, axis=0).tolist(),
            "p90_speed_normalized": np.quantile(norm_array, 0.9, axis=0).tolist(),
            "mean_angular_velocity": np.mean(curv_array, axis=0).tolist(),
            "p10_angular_velocity": np.quantile(curv_array, 0.1, axis=0).tolist(),
            "p90_angular_velocity": np.quantile(curv_array, 0.9, axis=0).tolist(),
            "mean_orig_angular_velocity": np.mean(orig_curv_array, axis=0).tolist(),
            "p10_orig_angular_velocity": np.quantile(orig_curv_array, 0.1, axis=0).tolist(),
            "p90_orig_angular_velocity": np.quantile(orig_curv_array, 0.9, axis=0).tolist(),
            "moves_used": int(norm_array.shape[0]),
        }
    else:
        profile = {
            "progress_0_1": [], "mean_speed_px_s": [], "mean_speed_normalized": [], 
            "p10_speed_normalized": [], "p90_speed_normalized": [], 
            "mean_angular_velocity": [], "p10_angular_velocity": [], "p90_angular_velocity": [],
            "mean_orig_angular_velocity": [], "p10_orig_angular_velocity": [], "p90_orig_angular_velocity": [],
            "moves_used": 0
        }

    return {"anti_macro": anti_macro, "average_move_profile": profile}


def _render_analysis_figure(analysis: dict[str, object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)
    fig.patch.set_facecolor("#f9fafb")

    anti_macro = analysis["anti_macro"]
    profile = analysis["average_move_profile"]

    ax_anti, ax_curve, ax_curv = axes

    # Simple straightness visualization (Anti-macro metrics left as-is)
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
    mean_orig_angular = profile.get("mean_orig_angular_velocity", [])
    p10_orig_angular = profile.get("p10_orig_angular_velocity", [])
    p90_orig_angular = profile.get("p90_orig_angular_velocity", [])

    if progress and mean_norm:
        ax_curve.plot(progress, mean_norm, color="#7c3aed", linewidth=2.0, label="mean normalized speed")
        ax_curve.fill_between(progress, p10_norm, p90_norm, color="#a78bfa", alpha=0.25, label="p10-p90 band")
        ax_curve.set_xlim(0.0, 1.0)
        ax_curve.set_ylim(0, 1.05) # PATCH: Ensure zero floor
        if mean_abs:
            ax_curve2 = ax_curve.twinx()
            ax_curve2.plot(progress, mean_abs, color="#ea580c", linewidth=1.1, alpha=0.7, label="mean absolute speed")
            ax_curve2.set_ylabel("mean speed (px/s)")
            ax_curve2.set_ylim(0, max(mean_abs) * 1.1 if mean_abs else 1.0) # PATCH: Fix negative padding
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

    if progress and mean_orig_angular:
        ax_curv.plot(progress, mean_orig_angular, color="#FF0000", linewidth=1.1, alpha=0.7, label="mean original heading angle")
        ax_curv.fill_between(progress, p10_orig_angular, p90_orig_angular, color="#ff6600", alpha=0.25, label="p10-p90 band")

    ax_curv.set_title("Average Heading Angle Shape Within One Move")
    ax_curv.set_xlabel("normalized move progress (0=start, 1=end)")
    ax_curv.set_ylabel("heading angle (rad)")
    ax_curv.legend()
    ax_curv.grid(alpha=0.2)

    fig.suptitle(f"Generated path analysis | moves analyzed={anti_macro.get('move_count', 0)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_parent = lambda p: p.parent.mkdir(parents=True, exist_ok=True)
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_fitts_plot(csv_path: Path, output_path: Path, W: float = 60.0) -> None:
    try:
        data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
        if data.size == 0:
            return
        x = data["distance"].astype(float)
        y = data["duration"].astype(float)
    except Exception:
        # fallback: try plain load
        try:
            arr = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
            if arr.size == 0:
                return
            if arr.ndim == 1 and arr.shape[0] >= 2:
                x = np.array([arr[0]])
                y = np.array([arr[1]])
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
    residuals = y - y_fit
    sigma = float(np.std(residuals))

    upper = y_fitts + sigma
    lower = y_fitts - sigma

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.scatter(x, y, label="Data", alpha=0.8)
    ax.plot(x_smooth, y_fitts, label="Fitts-like fit", linewidth=2)
    ax.fill_between(x_smooth, lower, upper, alpha=0.2, label="±1σ")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Duration")
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
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PATHS_CSV.exists():
        raise FileNotFoundError(
            f"Missing required path CSV: {PATHS_CSV}. Run plot_segmented.py first to generate it."
        )

    start_end_pairs = [("0copy", (200, 200), (400, 300))]*10 + load_start_end_pairs(PATHS_CSV)
    if not start_end_pairs:
        raise RuntimeError(f"No valid start/end rows found in {PATHS_CSV}")

    trajectories = []
    starts = []
    stroke_ids = []
    import time
    time_spent = []
    for i, (stroke_id, start, target) in enumerate(start_end_pairs):
        try:
            # curviness=1 default and curviness=0.1 ensures reasonably straight lines.
            st_time = time.perf_counter()
            traj = build_trajectory(start, target, target_width=TARGET_WIDTH, sample_rate=SAMPLE_RATE) #, initial_velocity=[401.08062239, 375.70308315], curviness=0.8)
            time_spent.append(time.perf_counter() - st_time)
            trajectories.append(traj)
            starts.append(start)
            stroke_ids.append(stroke_id)
            # print(f"Generated trajectory {i + 1} ({stroke_id}): {start} -> {target}")
        except Exception as e:
            raise e
            print(f"Error generating trajectory {i + 1}: {e}")

    if not trajectories:
        raise RuntimeError("No trajectories were generated from the provided start/end CSV rows.")

    analysis = _analyze_generated_trajectories(trajectories)
    _render_analysis_figure(analysis, ANALYSIS_OUTPUT)
    print(f"Saved {ANALYSIS_OUTPUT}")

    # Render Fitts-like plot from moves-human.csv
    _render_fitts_plot(MOVES_CSV, OUTPUT_DIR / Fitts_OUTPUT.name)

    print(f"Done. Images saved in: {OUTPUT_DIR.resolve()}")
    print(f"Average time per trajectory: {np.mean(time_spent):.8f} seconds")

    if SAVE_TRAJECTORIES:
        subdir = OUTPUT_DIR / "trajectories"
        subdir.mkdir(parents=True, exist_ok=True)
        for idx, traj in enumerate(trajectories[:MAX_TRAJECTORY_FIGURES]):
            start = starts[idx]
            stroke_id = stroke_ids[idx]
            target = traj["nominal_end"]
            out_name = f"path_{stroke_id}_{idx + 1:02d}.png"
            out_path = subdir / out_name
            save_trajectory_figure(traj, start, out_path, idx, stroke_id, SAMPLE_RATE)
            print(f"Saved {out_path}")
    
if __name__ == "__main__":
    main()
