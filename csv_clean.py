"""Basic cleaning for raw mouse movement CSV files."""
 
from __future__ import annotations
 
import argparse
from math import hypot
from pathlib import Path
 
import pandas as pd
 
# Tail trim settings.
TAIL_SPEED_THRESHOLD = 20.0
TAIL_DURATION_S      = 0.025

DEFAULT_INPUT_CSV          = "strokes_csv/raw_mouse_data.csv"
DEFAULT_OUTPUT_CSV         = "strokes_csv/raw_mouse_cleaned.csv"
DEFAULT_MIN_SPEED          = 100.0   # px/s
DEFAULT_MAX_SPEED          = 5000.0  # px/s
DEFAULT_MIN_PATH_DISTANCE  = 50.0    # px
DEFAULT_MAX_PATH_DISTANCE  = 20000.0
DEFAULT_MIN_DURATION_S     = 0.05
 
def trim_dead_tail(group: pd.DataFrame) -> pd.DataFrame:
    """
    Drop trailing move-rows that represent the mouse decelerating to a stop.
 
    We find the last move whose speed > TAIL_SPEED_THRESHOLD, then drop
    everything after it IF that trailing segment lasts >= TAIL_DURATION_S.
    Non-move rows (button_down/up, scroll) are never trimmed.
    """
    move_mask = group["event_type"] == "move"
    meaningful = group[move_mask & (group["speed"] > TAIL_SPEED_THRESHOLD)]
 
    if meaningful.empty:
        return group
 
    last_idx = meaningful.index[-1]
    last_ts  = group.loc[last_idx, "timestamp"]
 
    tail = group[group.index > last_idx]
    if tail.empty:
        return group
 
    tail_duration = float(tail["timestamp"].iloc[-1]) - float(last_ts)
    if tail_duration >= TAIL_DURATION_S:
        return group[group.index <= last_idx].copy()
 
    return group
 
 
def fix_boundary_speeds(group: pd.DataFrame) -> pd.DataFrame:
    """
    Zero speed and dt at the first and last move row of each stroke.
 
    The first point has no prior point to measure from, and the last point
    has non-zero arrival speed even though the mouse stops there. Both
    are artefacts of the delta-based speed calculation.
    """
    move_idx = group.index[group["event_type"] == "move"]
    if len(move_idx) == 0:
        return group
 
    for col in ("speed", "dt"):
        group.loc[move_idx[0],  col] = 0.0
        group.loc[move_idx[-1], col] = 0.0
 
    return group
 
 
def recompute_deltas(group: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate dt, dx, dy, speed for move rows from the (possibly trimmed)
    timestamps and coordinates so they are internally consistent.
 
    Uses the normalised 'timestamp_s' column (always seconds) added at load time.
    Non-move rows keep their original values (speed = 0 for those).
    """
    move_mask = group["event_type"] == "move"
    if move_mask.sum() < 2:
        return group
 
    g = group.copy()
 
    moves = g[move_mask].copy()
    moves["dt"]    = moves["timestamp_s"].diff().fillna(0.0)
    moves["dx"]    = moves["x"].diff().fillna(0).astype(int)
    moves["dy"]    = moves["y"].diff().fillna(0).astype(int)
    moves["speed"] = moves.apply(
        lambda r: hypot(r["dx"], r["dy"]) / r["dt"] if r["dt"] > 0 else 0.0,
        axis=1,
    )
 
    g.loc[move_mask, ["dt", "dx", "dy", "speed"]] = moves[["dt", "dx", "dy", "speed"]]
    return g
 
 
def resequence(group: pd.DataFrame) -> pd.DataFrame:
    """Reset seq to 0-based contiguous integers after any trimming."""
    group = group.copy()
    group["seq"] = range(len(group))
    return group
 
def dedupe_timestamps(group: pd.DataFrame) -> pd.DataFrame:
    """
    Drop move rows whose timestamp_s is identical to the previous move row.
 
    The recorder can emit two events at the same nanosecond tick (e.g. when
    the OS coalesces input).  A dt=0 move row makes spline knots degenerate
    (zero-length interval) and causes divide-by-zero in any downstream
    interpolation that uses the Catmull-Rom or natural-cubic formula
    c = dx1 / (dx2 * (dx1 + dx2)).
 
    Non-move rows (button, scroll) are never dropped.
    """
    move_mask = group["event_type"] == "move"
    if move_mask.sum() < 2:
        return group
 
    g = group.copy()
    # Within move rows, mark duplicates (same timestamp_s as previous move)
    move_ts = g.loc[move_mask, "timestamp_s"]
    is_dup = move_ts.duplicated(keep="first")          # True on the later copy
    g = g[~(move_mask & is_dup.reindex(g.index, fill_value=False))]
    return g
 
 
# Gap threshold: a dt this large on a non-move row signals the mouse was idle.
# The move row immediately *after* such a gap has a stale/huge dt value from
# recompute_deltas (because it diffs against the idle-period endpoint).
GAP_DT_THRESHOLD = 1.0   # seconds
 
 
def drop_post_gap_moves(group: pd.DataFrame) -> pd.DataFrame:
    """
    Drop move rows that immediately follow a large dt gap in the event stream.
 
    After recompute_deltas, a move row right after a button-hold or idle
    period inherits a dt equal to the length of that pause (e.g. 64 s).
    The computed speed for that row is near-zero even if the mouse was
    actually moving quickly, which corrupts spline parameterisation and
    arc-length integrals.
 
    We drop only the single 'first step after the gap' move row; the rest
    of the stroke is kept.
    """
    g = group.copy()
    move_mask = g["event_type"] == "move"
 
    # Rows (move or not) whose dt exceeds the threshold signal a gap
    big_gap = g["dt"] > GAP_DT_THRESHOLD
 
    # For each gap row, find the next move row and flag it for removal
    drop_idx: set = set()
    gap_positions = g.index[big_gap].tolist()
    for gap_pos in gap_positions:
        # Find the first move row that comes after this gap position
        candidates = g.loc[gap_pos:].index
        after_moves = candidates[candidates.isin(g.index[move_mask]) & (candidates != gap_pos)]
        if len(after_moves) > 0:
            drop_idx.add(after_moves[0])
 
    if drop_idx:
        g = g.drop(index=drop_idx)
    return g
 
 
# Per-stroke predicates.
 
def has_enough_move_rows(group: pd.DataFrame, min_rows: int = 2) -> bool:
    return (group["event_type"] == "move").sum() >= min_rows
 
 
def speed_in_range(group: pd.DataFrame, lo: float, hi: float) -> bool:
    peak = group["speed"].max()
    return lo <= peak <= hi
 
 
def path_distance_ok(group: pd.DataFrame, min_dist: float, max_dist: float) -> bool:
    moves = group[group["event_type"] == "move"]
    dist  = ((moves["x"].diff() ** 2 + moves["y"].diff() ** 2) ** 0.5).sum()
    return dist >= min_dist and dist <= max_dist
 
 
def duration_ok(group: pd.DataFrame, min_dur: float) -> bool:
    moves = group[group["event_type"] == "move"]
    if len(moves) < 2:
        return False
    return float(moves["timestamp_s"].iloc[-1]) - float(moves["timestamp_s"].iloc[0]) >= min_dur
 
 
# Heuristic thresholds for median timestamp value:
#   > 1e15: nanoseconds
#   > 1e12: microseconds
#   > 1e9: milliseconds
#   else: seconds
 
def _detect_ts_scale(series: pd.Series) -> float:
    """Return divisor to convert the timestamp column to seconds."""
    med = series.median()
    if med > 1e15:
        return 1e9
    if med > 1e12:
        return 1e6
    if med > 1e9:
        return 1e3
    return 1.0
 
 
def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'timestamp_s' column that is always in relative seconds.
 
    Detects the unit of 'timestamp' automatically, converts to seconds,
    then offsets to zero from the first event so dt is meaningful.
    """
    scale = _detect_ts_scale(df["timestamp"])
    ts_s = df["timestamp"] / scale
 
    # Convert epoch-based seconds to a relative offset.
    if ts_s.median() > 1e8:
        ts_s = ts_s - ts_s.iloc[0]
 
    df = df.copy()
    df["timestamp_s"] = ts_s
    if scale != 1.0:
        print(f"  [normalize] timestamp detected as x{scale:.0e}; "
              f"converted to relative seconds (timestamp_s).")
    return df
 
 
def clean_csv(
    input_path:       Path,
    output_path:      Path,
    min_speed:        float,
    max_speed:        float,
    min_path_dist:    float,
    max_path_dist:    float,
    min_duration_s:   float,
    do_trim_tail:     bool,
    do_fix_speeds:    bool,
    do_recompute:     bool,
) -> None:
    df = pd.read_csv(input_path)
 
    required = {"stroke_id", "seq", "timestamp", "x", "y",
                "dt", "dx", "dy", "speed", "event_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
 
    df = normalize_timestamps(df)
 
    # Keep only move events for path-distance and speed stats.
    df = df[df["event_type"] == "move"].copy()
 
    strokes_in = df["stroke_id"].nunique()
    rows_in    = len(df)
 
    processed: list[pd.DataFrame] = []
 
    for _, group in df.groupby("stroke_id", sort=False):
        g = group.copy()
 
        if do_trim_tail:
            g = trim_dead_tail(g)
 
        if do_recompute:
            g = recompute_deltas(g)
 
        # Always run these after recompute so dt values are trustworthy
        # g = drop_post_gap_moves(g)
        g = dedupe_timestamps(g)
 
        if do_fix_speeds:
            g = fix_boundary_speeds(g)
 
        if not has_enough_move_rows(g):
            continue
        if not speed_in_range(g, min_speed, max_speed):
            continue
        if not path_distance_ok(g, min_path_dist, max_path_dist):
            continue
        if not duration_ok(g, min_duration_s):
            continue
 
        g = resequence(g)
        processed.append(g)
 
    if processed:
        out = pd.concat(processed, ignore_index=True)
        old_ids   = out["stroke_id"].unique()
        id_map    = {old: new for new, old in enumerate(old_ids)}
        out["stroke_id"] = out["stroke_id"].map(id_map)
    else:
        out = df.iloc[:0].copy()   # empty frame with correct columns
 
    out = out.drop(columns=["timestamp_s"], errors="ignore")
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
 
    strokes_out = out["stroke_id"].nunique() if len(out) else 0
    rows_out    = len(out)
 
    print(f"Input : {input_path}  ({rows_in:,} rows, {strokes_in} strokes)")
    print(f"Output: {output_path}  ({rows_out:,} rows, {strokes_out} strokes)")
    print()
    print(f"  Strokes removed : {strokes_in - strokes_out}"
          f"  ({100*(strokes_in-strokes_out)/max(strokes_in,1):.1f}%)")
    print(f"  Rows removed    : {rows_in - rows_out}"
          f"  ({100*(rows_in-rows_out)/max(rows_in,1):.1f}%)")
    print()
    print(f"  Filters applied:")
    print(f"    tail trim         : {'yes' if do_trim_tail else 'no'}"
          f"  (threshold {TAIL_SPEED_THRESHOLD} px/s, min tail {TAIL_DURATION_S}s)")
    print(f"    boundary speed fix: {'yes' if do_fix_speeds else 'no'}")
    print(f"    delta recompute   : {'yes' if do_recompute else 'no'}")
    print(f"    speed range       : [{min_speed}, {max_speed}] px/s")
    print(f"    min path distance : {min_path_dist} px")
    print(f"    min duration      : {min_duration_s} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic clean and standardize mouse movement CSV.")
    parser.add_argument("--input_csv", help="Input CSV file", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", help="Output cleaned CSV file", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--min-speed", type=float, default=DEFAULT_MIN_SPEED)
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED)
    parser.add_argument("--min-path-dist", type=float, default=DEFAULT_MIN_PATH_DISTANCE)
    parser.add_argument("--min-duration",      type=float, default=DEFAULT_MIN_DURATION_S,
                   help="Minimum stroke duration in seconds (default: 0.05)")
 
    parser.add_argument("--no-trim-tail",     action="store_true",
                   help="Disable dead-tail trimming")
    parser.add_argument("--no-fix-speeds",    action="store_true",
                   help="Disable zeroing of first/last move speed")
    parser.add_argument("--no-recompute",     action="store_true",
                   help="Disable recomputing dt/dx/dy/speed after trimming")

    args = parser.parse_args()

    clean_csv(
        input_path=Path(args.input_csv),
        output_path=Path(args.output_csv),
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        min_path_dist=args.min_path_dist,
        max_path_dist=DEFAULT_MAX_PATH_DISTANCE,
        min_duration_s = args.min_duration,
        do_trim_tail   = not args.no_trim_tail,
        do_fix_speeds  = not args.no_fix_speeds,
        do_recompute   = not args.no_recompute,
    )


if __name__ == "__main__":
    main()
