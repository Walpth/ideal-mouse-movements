#!/usr/bin/env python3
"""
log_movements.py

Logs mouse motion / buttons / scroll from evdev into a CSV with columns:

stroke_id,seq,timestamp,x,y,dt,dx,dy,speed,button_held,event_type,button_id,scroll_dy

Run:
    python3 log_movements.py
or from VS Code with the integrated terminal.

Notes:
- Does NOT call dev.grab(), so your mouse remains usable.
- Uses evdev packet reconstruction via SYN_REPORT.
- timestamp is written as nanoseconds since Unix epoch.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from evdev import InputDevice, ecodes, list_devices


# Defaults you can change directly.
OUTPUT_CSV = "strokes_csv/raw_mouse_data.csv"
DEVICE_PATH = None          # e.g. "/dev/input/event5"; set to None for auto-detect
STROKE_IDLE_MS = 200        # start a new stroke after this much idle time
AUTO_START_NEW_STROKE_ON_PRESS = True


def is_mouse_device(dev: InputDevice) -> bool:
    caps = dev.capabilities(absinfo=False)
    rel = set(caps.get(ecodes.EV_REL, []))
    key = set(caps.get(ecodes.EV_KEY, []))
    return (
        ecodes.REL_X in rel
        and ecodes.REL_Y in rel
        and any(k in key for k in (ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE))
    )


def find_mouse_device_path() -> str:
    for path in list_devices():
        try:
            dev = InputDevice(path)
            if is_mouse_device(dev):
                return path
        except Exception:
            pass
    raise RuntimeError("No likely mouse device found. Set DEVICE_PATH manually.")


def event_ts_ns(ev) -> int:
    return int(ev.sec * 1_000_000_000 + ev.usec * 1_000)


def button_name(code: int) -> str:
    mapping = {
        ecodes.BTN_LEFT: "BTN_LEFT",
        ecodes.BTN_RIGHT: "BTN_RIGHT",
        ecodes.BTN_MIDDLE: "BTN_MIDDLE",
        ecodes.BTN_SIDE: "BTN_SIDE",
        ecodes.BTN_EXTRA: "BTN_EXTRA",
        ecodes.BTN_FORWARD: "BTN_FORWARD",
        ecodes.BTN_BACK: "BTN_BACK",
    }
    return mapping.get(code, str(code))


def main() -> int:
    device_path = DEVICE_PATH or find_mouse_device_path()
    dev = InputDevice(device_path)

    print(f"Using device: {dev.path} ({dev.name})")
    print("Writing CSV to:", OUTPUT_CSV)
    print("Press Ctrl-C to stop.\n")

    out_path = Path(OUTPUT_CSV)

    # Current reconstructed position.
    x = 0
    y = 0

    # Packet accumulators for motion.
    packet_dx = 0
    packet_dy = 0
    packet_ts_ns = None

    # Output sequencing.
    seq = 0
    stroke_id = 0

    # State for timing / buttons.
    last_logged_ts_ns = None
    last_motion_ts_ns = None
    button_held = 0

    def next_stroke_if_needed(ts_ns: int):
        nonlocal stroke_id, last_motion_ts_ns
        if last_motion_ts_ns is None:
            return
        idle_ms = (ts_ns - last_motion_ts_ns) / 1_000_000.0
        if idle_ms >= STROKE_IDLE_MS:
            stroke_id += 1

    def write_row(writer, ts_ns: int, dx: int, dy: int, event_type: str, button_id: str = "", scroll_dy: int = 0):
        nonlocal seq, last_logged_ts_ns

        dt_ms = 0.0 if last_logged_ts_ns is None else (ts_ns - last_logged_ts_ns) / 1_000_000.0
        speed = math.hypot(dx, dy) / (dt_ms / 1000.0) if dt_ms > 0 else 0.0

        seq += 1
        writer.writerow([
            stroke_id,
            seq,
            ts_ns,
            x,
            y,
            f"{dt_ms:.3f}",
            dx,
            dy,
            f"{speed:.3f}",
            button_held,
            event_type,
            button_id,
            scroll_dy,
        ])
        last_logged_ts_ns = ts_ns

    try:
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stroke_id", "seq", "timestamp", "x", "y", "dt", "dx", "dy",
                "speed", "button_held", "event_type", "button_id", "scroll_dy"
            ])

            for ev in dev.read_loop():
                ts_ns = event_ts_ns(ev)

                if ev.type == ecodes.EV_REL:
                    if ev.code == ecodes.REL_X:
                        packet_dx += ev.value
                        packet_ts_ns = ts_ns
                    elif ev.code == ecodes.REL_Y:
                        packet_dy += ev.value
                        packet_ts_ns = ts_ns
                    elif ev.code in (ecodes.REL_WHEEL, ecodes.REL_WHEEL_HI_RES):
                        # Flush any pending motion first.
                        if packet_dx != 0 or packet_dy != 0 and packet_ts_ns is not None:
                            next_stroke_if_needed(packet_ts_ns)
                            x += packet_dx
                            y += packet_dy
                            write_row(writer, packet_ts_ns, packet_dx, packet_dy, "move")
                            last_motion_ts_ns = packet_ts_ns
                            packet_dx = 0
                            packet_dy = 0
                            packet_ts_ns = None

                        if stroke_id == 0:
                            stroke_id = 1
                        write_row(writer, ts_ns, 0, 0, "scroll", scroll_dy=ev.value)

                    elif ev.code in (ecodes.REL_HWHEEL, ecodes.REL_HWHEEL_HI_RES):
                        if stroke_id == 0:
                            stroke_id = 1
                        write_row(writer, ts_ns, 0, 0, "scroll", scroll_dy=ev.value)

                elif ev.type == ecodes.EV_KEY:
                    # Flush any pending motion before logging a button event.
                    if packet_dx != 0 or packet_dy != 0:
                        flush_ts = packet_ts_ns if packet_ts_ns is not None else ts_ns
                        next_stroke_if_needed(flush_ts)
                        x += packet_dx
                        y += packet_dy
                        write_row(writer, flush_ts, packet_dx, packet_dy, "move")
                        last_motion_ts_ns = flush_ts
                        packet_dx = 0
                        packet_dy = 0
                        packet_ts_ns = None

                    if ev.code in (
                        ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE,
                        ecodes.BTN_SIDE, ecodes.BTN_EXTRA, ecodes.BTN_FORWARD, ecodes.BTN_BACK,
                    ):
                        if ev.value == 1:
                            if AUTO_START_NEW_STROKE_ON_PRESS:
                                stroke_id += 1
                            button_held = 1
                        elif ev.value == 0:
                            button_held = 0

                        if stroke_id == 0:
                            stroke_id = 1
                        write_row(writer, ts_ns, 0, 0, "button", button_id=button_name(ev.code))

                elif ev.type == ecodes.EV_SYN and ev.code == ecodes.SYN_REPORT:
                    # Finalize the accumulated motion packet.
                    if packet_dx != 0 or packet_dy != 0:
                        flush_ts = packet_ts_ns if packet_ts_ns is not None else ts_ns
                        next_stroke_if_needed(flush_ts)
                        x += packet_dx
                        y += packet_dy
                        write_row(writer, flush_ts, packet_dx, packet_dy, "move")
                        last_motion_ts_ns = flush_ts
                        packet_dx = 0
                        packet_dy = 0
                        packet_ts_ns = None

    except KeyboardInterrupt:
        pass
    finally:
        dev.close()

    print(f"Saved: {os.path.abspath(OUTPUT_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
