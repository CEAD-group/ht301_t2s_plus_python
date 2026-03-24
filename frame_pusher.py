#!/usr/bin/env python3
"""Push processed thermal camera frames to a v4l2 virtual video device.

Reads from the T2S+/HT-301 thermal camera, applies colormap and contrast
enhancement, and pipes BGR24 frames to /dev/video10 via ffmpeg.

Requires:
  - v4l2loopback kernel module loaded (creates /dev/video10)
  - ffmpeg installed

Usage:
  python frame_pusher.py [--device /dev/video10] [--orientation 0]
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

import irpythermal


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


def main():
    parser = argparse.ArgumentParser(description="Push thermal frames to v4l2 virtual device")
    parser.add_argument("--device", default="/dev/video10", help="v4l2 loopback device path")
    parser.add_argument("--orientation", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    camera = irpythermal.Camera()
    width, height = camera.width, camera.height
    print(f"Camera: {width}x{height} @ {args.fps}fps -> {args.device}")

    if not os.path.exists(args.device):
        print(f"Error: {args.device} not found. Is v4l2loopback loaded?")
        sys.exit(1)

    ffmpeg_cmd = [
        "ffmpeg",
        "-re",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(args.fps),
        "-i", "-",
        "-f", "v4l2",
        args.device,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame = cv2.equalizeHist(frame)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
            frame = increase_luminance_contrast(frame)
            frame = rotate_frame(frame, args.orientation)

            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("ffmpeg pipe broken, exiting")
                break

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()


if __name__ == "__main__":
    main()
