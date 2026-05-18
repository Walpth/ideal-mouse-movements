#!/usr/bin/env python3
"""Train a compact multi-resolution mouse movement model.

Curve representation: arc-length parameterised cumulative heading θ(s).
Speed representation: arc-length parameterised speed v(s).
Joint Model: registration-light functional Joint-PCA (FPCA).
Latent Model: Gaussian Mixture Model (GMM) on Joint-PCA scores conditioned on features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _relative_seconds(series: pd.Series | np.ndarray) -> np.ndarray:
    t = np.asarray(series, dtype=float)
    scale = 1e-9 if len(t) and t[0] > 1e12 else 1.0
    return (t - t[0]) * scale


def plot_sample_profiles(
    raw_samples: list[dict],
    n_samples: int = 5,
    seed: int = 42,
):
    import matplotlib.pyplot as plt

    if len(raw_samples) == 0:
        print("No samples available for plotting.")
        return

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(raw_samples), size=min(n_samples, len(raw_samples)), replace=False)

    plt.figure(figsize=(10, 8))

    for i, idx in enumerate(indices):
        sample = raw_samples[idx]

        # Recover normalized speed from stored asinh(speed)
        speed = np.sinh(sample["v_asinh"])

        # Theta profile
        theta = sample["theta"]

        # Use normalized phase axis [0,1]
        t = np.linspace(0.0, 1.0, len(speed))

        # Speed plot
        plt.subplot(2, 1, 1)
        plt.plot(t, speed, label=f"Stroke {idx}")
        plt.ylabel("Normalized Speed")
        plt.title("Training Stroke Profiles")
        plt.grid(True)

        # Theta plot
        plt.subplot(2, 1, 2)
        plt.plot(t, theta)
        plt.xlabel("Normalized Arc-Length Phase")
        plt.ylabel("Theta (rad)")
        plt.grid(True)

    plt.subplot(2, 1, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()


def _resample(values_x: np.ndarray, values_y: np.ndarray, grid_x: np.ndarray) -> np.ndarray:
    order = np.argsort(values_x)
    vx, vy = values_x[order], values_y[order]
    vx, idx = np.unique(vx, return_index=True)
    return np.interp(grid_x, vx, vy[idx], left=float(vy[idx][0]), right=float(vy[idx][-1]))


def _decompose_hierarchical(signal: np.ndarray, level_sizes: list[int]) -> list[np.ndarray]:
    coeffs = []
    prev_grid = np.linspace(0.0, 1.0, level_sizes[0])
    curr_vals = _resample(np.linspace(0.0, 1.0, len(signal)), signal, prev_grid)
    coeffs.append(curr_vals.astype(np.float32))
    for next_n in level_sizes[1:]:
        next_grid = np.linspace(0.0, 1.0, next_n)
        true_vals = _resample(np.linspace(0.0, 1.0, len(signal)), signal, next_grid)
        prev_up = _resample(prev_grid, curr_vals, next_grid)
        coeffs.append((true_vals[1::2] - prev_up[1::2]).astype(np.float32))
        prev_grid, curr_vals = next_grid, true_vals
    return coeffs


def compute_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points))
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    ds = np.maximum(ds, 1e-12)
    return np.concatenate([[0.0], np.cumsum(ds)])

def compute_theta_arclength(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 5: return np.linspace(0.0, 1.0, len(points)), np.zeros(len(points))
    smoothed = savgol_filter(points, 11, 3, axis=0, mode="mirror")
    ds = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_norm = s / max(s[-1], 1e-9)
    dx, dy = np.gradient(smoothed[:, 0], s), np.gradient(smoothed[:, 1], s)
    theta = np.unwrap(np.arctan2(dy, dx))
    theta -= np.linspace(theta[0], theta[-1], len(theta)) # Detrend
    return s_norm, (theta - theta[0]).astype(np.float32)

def compute_raw_theta_arclength(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        return np.linspace(0.0, 1.0, len(points)), np.zeros(len(points))
    diffs = np.diff(points, axis=0, prepend=points[:1])
    step = np.linalg.norm(diffs, axis=1)
    heading = np.arctan2(diffs[:, 1], diffs[:, 0])
    moving = np.where(step > 0.0)[0]
    if len(moving) == 0:
        return np.linspace(0.0, 1.0, len(points)), np.zeros(len(points))
    heading[: moving[0] + 1] = heading[moving[0]]
    for i in range(moving[0] + 1, len(heading)):
        if step[i] <= 0.0:
            heading[i] = heading[i - 1]
    s = np.cumsum(step)
    s_norm = s / max(float(s[-1]), 1e-9)
    theta = np.unwrap(heading)
    theta -= np.linspace(theta[0], theta[-1], len(theta))
    theta -= theta[0]
    return s_norm, theta.astype(np.float32)

def _highpass_heading_residual(raw_theta: np.ndarray, smooth_theta: np.ndarray) -> np.ndarray:
    residual = np.asarray(raw_theta, dtype=float) - np.asarray(smooth_theta, dtype=float)
    if len(residual) >= 17:
        trend = savgol_filter(residual, 17, 2, mode="mirror")
        residual = residual - trend
    elif len(residual) >= 5:
        trend = savgol_filter(residual, len(residual) if len(residual) % 2 else len(residual) - 1, 2, mode="mirror")
        residual = residual - trend
    residual -= np.linspace(residual[0], residual[-1], len(residual))
    return residual.astype(np.float32)


def detect_curvature_landmarks(
    s_norm: np.ndarray,
    theta: np.ndarray,
    n_landmarks: int = 2,
    min_prominence_ratio: float = 0.15,
) -> np.ndarray:
    if len(theta) < 5 or n_landmarks == 0:
        return np.array([], dtype=float)

    curvature = np.gradient(theta, s_norm)
    abs_curv = np.abs(curvature)
    max_curv = float(abs_curv.max())
    if max_curv < 1e-6:
        return np.array([], dtype=float)

    min_dist = max(3, len(s_norm) // 8)
    peaks, props = find_peaks(
        abs_curv,
        prominence=min_prominence_ratio * max_curv,
        distance=min_dist,
    )
    if len(peaks) == 0:
        return np.array([], dtype=float)

    top_idx = np.argsort(props["prominences"])[::-1][:n_landmarks]
    selected = np.sort(peaks[top_idx])
    return s_norm[selected]

def compute_reference_landmarks(
    all_landmarks: list[np.ndarray],
    n_landmarks: int,
) -> np.ndarray:
    ref: list[float] = []
    for k in range(n_landmarks):
        positions = [float(lm[k]) for lm in all_landmarks if len(lm) > k]
        if positions:
            ref.append(float(np.median(positions)))
    return np.array(ref, dtype=float)

def register_stroke(
    s_norm: np.ndarray,
    theta: np.ndarray,
    speed: np.ndarray,
    stroke_landmarks: np.ndarray,
    reference_landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uniform_s = np.linspace(0.0, 1.0, len(s_norm))
    n_lm = min(len(stroke_landmarks), len(reference_landmarks))
    
    if n_lm == 0:
        return np.interp(uniform_s, s_norm, theta), np.interp(uniform_s, s_norm, speed)

    src = np.concatenate([[0.0], stroke_landmarks[:n_lm], [1.0]])
    dst = np.concatenate([[0.0], reference_landmarks[:n_lm], [1.0]])

    warped_s = np.interp(s_norm, src, dst)
    theta_registered = np.interp(uniform_s, warped_s, theta)
    speed_registered = np.interp(uniform_s, warped_s, speed)
    
    return theta_registered, speed_registered


def _iter_move_substrokes(df: pd.DataFrame):
    if {"substroke", "event_type"}.issubset(df.columns):
        groups = df[df["event_type"] == "move"].groupby(["stroke_id", "substroke"], sort=False)
    else:
        groups = df.groupby("stroke_id", sort=False)
    for _, group in groups:
        yield group.sort_values("seq").drop_duplicates(subset=["timestamp"]).copy()


def _extract_layout_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    samples: list[dict[str, object]] = []

    for _, stroke in df.groupby("stroke_id", sort=False):
        stroke = stroke.sort_values("seq").reset_index(drop=True)
        move_groups = [
            g[g["event_type"] == "move"].copy()
            for _, g in stroke.groupby("substroke", sort=False)
            if (g["event_type"] == "move").any()
        ]
        if not move_groups:
            continue

        first_move, last_move = move_groups[0], move_groups[-1]
        stroke_t = _relative_seconds(stroke["timestamp"])
        total_duration = max(float(stroke_t[-1] - stroke_t[0]), 1e-3)
        start = first_move[["x", "y"]].iloc[0].to_numpy(dtype=float)
        end = last_move[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        chord = end - start
        chord_len = max(float(np.linalg.norm(chord)), 1.0)
        normal = np.array([-chord[1], chord[0]], dtype=float) / chord_len

        move_durations = []
        move_distances = []
        waypoints = []
        for i, move in enumerate(move_groups):
            mt = _relative_seconds(move["timestamp"])
            move_durations.append(max(float(mt[-1] - mt[0]), 1e-4))
            p0 = move[["x", "y"]].iloc[0].to_numpy(dtype=float)
            p1 = move[["x", "y"]].iloc[-1].to_numpy(dtype=float)
            move_distances.append(max(float(np.linalg.norm(p1 - p0)), 1.0))
            if i < len(move_groups) - 1:
                rel = p1 - start
                along = float(np.dot(rel, chord) / (chord_len * chord_len))
                perp = float(np.dot(rel, normal) / chord_len)
                waypoints.append((along, perp))

        stops = stroke[stroke["event_type"] == "stop"]["dt"].to_numpy(dtype=float)
        n = len(move_groups)
        total_move_duration = max(float(sum(move_durations)), 1e-4)
        total_move_distance = max(float(sum(move_distances)), 1.0)

        samples.append({
            "features": (total_duration, chord_len),
            "n": n,
            "move_dur_frac": np.asarray(move_durations, dtype=float) / total_move_duration,
            "move_dist_frac": np.asarray(move_distances, dtype=float) / total_move_distance,
            "pause_frac": stops[: max(0, n - 1)] / total_duration,
            "waypoints": np.asarray(waypoints, dtype=float),
        })

    if not samples:
        return {}

    max_n = max(int(s["n"]) for s in samples)
    n_samples = len(samples)
    move_dur = np.full((n_samples, max_n), np.nan, dtype=np.float32)
    move_dist = np.full((n_samples, max_n), np.nan, dtype=np.float32)
    pause = np.full((n_samples, max(1, max_n - 1)), np.nan, dtype=np.float32)
    waypoint_along = np.full((n_samples, max(1, max_n - 1)), np.nan, dtype=np.float32)
    waypoint_perp = np.full((n_samples, max(1, max_n - 1)), np.nan, dtype=np.float32)

    features = np.asarray([s["features"] for s in samples], dtype=np.float32)
    counts = np.asarray([s["n"] for s in samples], dtype=np.int16)
    for i, sample in enumerate(samples):
        n = int(sample["n"])
        move_dur[i, :n] = sample["move_dur_frac"]
        move_dist[i, :n] = sample["move_dist_frac"]
        if n > 1:
            p = np.asarray(sample["pause_frac"], dtype=float)
            pause[i, : len(p)] = p
            w = np.asarray(sample["waypoints"], dtype=float)
            waypoint_along[i, : len(w)] = w[:, 0]
            waypoint_perp[i, : len(w)] = w[:, 1]

    log_features = np.log(np.column_stack([np.maximum(features[:, 0], 1e-3), np.maximum(features[:, 1], 1.0)]))
    return {
        "layout_features": features,
        "layout_log_feat_mean": log_features.mean(axis=0),
        "layout_log_feat_std": np.maximum(log_features.std(axis=0), 1e-6),
        "layout_n_submoves": counts,
        "layout_move_dur_frac": move_dur,
        "layout_move_dist_frac": move_dist,
        "layout_pause_frac": pause,
        "layout_waypoint_along": waypoint_along,
        "layout_waypoint_perp": waypoint_perp,
    }


def train_model(csv_path: Path, output_model: Path, plot: bool = False):
    df = pd.read_csv(csv_path)
    level_sizes = [9, 17, 33, 65, 129]
    level_comp =  [6, 8, 16, 28, 16]
    phase_grid = np.linspace(0.0, 1.0, 129)
    
    raw_samples = []
    theta_residuals = []

    for stroke in _iter_move_substrokes(df):
        if len(stroke) < 10: continue
            
        t, v, xy = stroke["timestamp"].values, stroke["speed"].values, stroke[["x","y"]].values
        t_raw = _relative_seconds(t)
        dur = max(t_raw[-1] - t_raw[0], 1e-3)
        dist = max(np.linalg.norm(xy[-1] - xy[0]), 1.0) # Prevent 0 dist
        
        v_avg = dist / dur
        v_norm = v / v_avg
        v_asinh = np.arcsinh(v_norm) 
        
        s_norm, theta = compute_theta_arclength(xy)
        theta_grid = _resample(s_norm, theta, phase_grid)
        raw_s_norm, raw_theta = compute_raw_theta_arclength(xy)
        raw_theta_grid = _resample(raw_s_norm, raw_theta, phase_grid)
        theta_residuals.append(_highpass_heading_residual(raw_theta_grid, theta_grid))
        raw_samples.append({
            "v_asinh": _resample(s_norm, v_asinh, phase_grid), 
            "theta": theta_grid,
            "feat": [dur, dist]
        })

    if not raw_samples:
        raise RuntimeError(f"No movement substrokes found in {csv_path}")

    if plot:
        plot_sample_profiles(raw_samples, n_samples=10)
    
    features = np.array([s["feat"] for s in raw_samples])
    feat_mean, feat_std = np.mean(features, axis=0), np.std(features, axis=0)
    feat_std[feat_std == 0] = 1.0
    features_norm = (features - feat_mean) / feat_std

    residual_mat = np.stack(theta_residuals)
    theta_residual_std = 1.4826 * np.median(np.abs(residual_mat - np.median(residual_mat, axis=0)), axis=0)
    theta_residual_std = np.clip(theta_residual_std, 0.0, 0.09).astype(np.float32)
    theta_residual_std[:4] = 0.0
    theta_residual_std[-4:] = 0.0
    centered = residual_mat - residual_mat.mean(axis=1, keepdims=True)
    if centered.shape[1] > 1:
        num = float(np.sum(centered[:, 1:] * centered[:, :-1]))
        den = float(np.sum(centered[:, :-1] ** 2))
        theta_residual_corr = float(np.clip(num / den, 0.0, 0.95)) if den > 1e-9 else 0.35
    else:
        theta_residual_corr = 0.35

    arrays = {"config_json": json.dumps({"version": 14, "level_sizes": level_sizes}), 
              "feat_mean": feat_mean, "feat_std": feat_std}
    arrays["theta_residual_std"] = theta_residual_std
    arrays["theta_residual_corr"] = np.array([theta_residual_corr], dtype=np.float32)
    arrays.update(_extract_layout_arrays(df))

    for lvl in range(len(level_sizes)):
        s_mat = np.stack([_decompose_hierarchical(s["v_asinh"], level_sizes)[lvl] for s in raw_samples])
        c_mat = np.stack([_decompose_hierarchical(s["theta"], level_sizes)[lvl] for s in raw_samples])
        
        s_std, c_std = max(np.std(s_mat), 1e-4), max(np.std(c_mat), 1e-4)
        joint = np.concatenate([s_mat / s_std, c_mat / c_std], axis=1)
        
        pca = PCA(n_components=min(level_comp[lvl], joint.shape[0]-1)).fit(joint)
        scores = pca.transform(joint)
        
        # reg_covar is KEY here to prevent "Exploding Inverses" later
        gmm = GaussianMixture(n_components=10, covariance_type='full', reg_covar=1e-3, max_iter=300, n_init=2).fit(
            np.concatenate([features_norm, scores], axis=1)
        )

        arrays.update({
            f"l{lvl}_m": pca.mean_, f"l{lvl}_c": pca.components_,
            f"l{lvl}_gmm_w": gmm.weights_, f"l{lvl}_gmm_mu": gmm.means_, 
            f"l{lvl}_gmm_cov": gmm.covariances_,
            f"l{lvl}_scale": np.array([s_std, c_std]),
            f"l{lvl}_score_std": np.std(scores, axis=0) # Save for clamping
        })
    
    np.savez_compressed(output_model, **arrays)
    print(f"Model v14 trained on {len(raw_samples)} movement substrokes. Layout model integrated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the mouse movement profile and layout model.")
    parser.add_argument("csv_path", nargs="?", default="strokes_csv/mouse_segmented.csv")
    parser.add_argument("--output", default="main_pipeline/model.npz")
    parser.add_argument("--plot", action="store_true", help="Show sampled training profiles")
    args = parser.parse_args()
    train_model(Path(args.csv_path), Path(args.output), plot=args.plot)
