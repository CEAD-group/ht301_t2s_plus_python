#!/usr/bin/env python3
"""Push processed thermal camera frames to a v4l2 virtual video device.

Reads from the T2S+/HT-301 thermal camera, applies colormap and contrast
enhancement, and pipes BGR24 frames to /dev/video10 via ffmpeg.

Only processes and pushes frames when a consumer is reading from the
loopback device. The camera is always kept open to avoid USB reconnect
overhead; frames are simply discarded when idle.

Requires:
  - v4l2loopback kernel module loaded (creates /dev/video10)
  - ffmpeg installed
  - fuser command available (from psmisc package)

Usage:
  python frame_pusher.py [--device /dev/video10] [--orientation 0]
"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

import irpythermal


def has_consumers(device_path):
    """Check if any process other than our ffmpeg is reading the v4l2 device."""
    try:
        result = subprocess.run(
            ["fuser", device_path],
            capture_output=True, timeout=2,
        )
        # fuser prints PIDs to stderr; exit code 0 means at least one process
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If fuser is unavailable or hangs, assume consumers exist (safe default)
        return True


def increase_luminance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def rotate_frame(frame, orientation):
    if orientation == 90:
        return np.rot90(frame).copy()
    elif orientation == 180:
        return np.rot90(frame, 2).copy()
    elif orientation == 270:
        return np.rot90(frame, 3).copy()
    return frame


def _start_ffmpeg(width, height, fps, device):
    """Start an ffmpeg process that pipes raw BGR24 frames to v4l2 loopback."""
    cmd = [
        "ffmpeg",
        "-re",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-f", "v4l2",
        device,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _stop_ffmpeg(proc):
    """Gracefully stop the ffmpeg process."""
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Push thermal frames to v4l2 virtual device")
    parser.add_argument("--device", default="/dev/video10", help="v4l2 loopback device path")
    parser.add_argument("--orientation", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between consumer checks when idle")
    args = parser.parse_args()

    camera = irpythermal.Camera()
    width, height = camera.width, camera.height
    print(f"Camera: {width}x{height} @ {args.fps}fps -> {args.device}")

    if not os.path.exists(args.device):
        print(f"Error: {args.device} not found. Is v4l2loopback loaded?")
        sys.exit(1)

    proc = None
    streaming = False
    last_consumer_check = 0.0
    consumer_check_interval = args.poll_interval

    try:
        while True:
            # Always read from the camera to keep the USB device active
            # and avoid stale frames when streaming resumes.
            ret, frame = camera.read()
            if not ret:
                continue

            # Periodically check for consumers on the loopback device.
            now = time.monotonic()
            if now - last_consumer_check >= consumer_check_interval:
                last_consumer_check = now
                consumers = has_consumers(args.device)

                if consumers and not streaming:
                    print(f"Consumer detected on {args.device}, starting stream")
                    proc = _start_ffmpeg(width, height, args.fps, args.device)
                    streaming = True
                elif not consumers and streaming:
                    print(f"No consumers on {args.device}, pausing stream")
                    _stop_ffmpeg(proc)
                    proc = None
                    streaming = False

            if not streaming:
                continue

            # Process and push the frame.
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame = cv2.equalizeHist(frame)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
            frame = increase_luminance_contrast(frame)
            frame = rotate_frame(frame, args.orientation)

            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("ffmpeg pipe broken, restarting on next consumer check")
                _stop_ffmpeg(proc)
                proc = None
                streaming = False

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera.release()
        _stop_ffmpeg(proc)


if __name__ == "__main__":
    main()
