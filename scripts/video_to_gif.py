#!/usr/bin/env python3
"""Convert a local video file to a GIF using ffmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a local video to a GIF.")
    parser.add_argument("input", type=Path, help="Input video path, for example logs/run/videos/play.mp4.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to the input path with a .gif suffix.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Output GIF frames per second.")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Output GIF width in pixels. Use 0 to keep the original video width.",
    )
    parser.add_argument("--start", default=None, help="Optional start timestamp, for example 3 or 00:00:03.5.")
    parser.add_argument("--duration", default=None, help="Optional clip duration, for example 5 or 00:00:05.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output GIF if it already exists.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path.")
    return parser.parse_args()


def require_ffmpeg(ffmpeg: str) -> None:
    if shutil.which(ffmpeg) is None:
        raise SystemExit(
            f"Could not find '{ffmpeg}'. Install ffmpeg or pass the executable path with --ffmpeg."
        )


def build_base_input_args(args: argparse.Namespace) -> list[str]:
    input_args: list[str] = []
    if args.start:
        input_args.extend(["-ss", args.start])
    input_args.extend(["-i", str(args.input)])
    if args.duration:
        input_args.extend(["-t", args.duration])
    return input_args


def build_filter(fps: int, width: int) -> str:
    filters = [f"fps={fps}"]
    if width > 0:
        filters.append(f"scale={width}:-1:flags=lanczos")
    return ",".join(filters)


def run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def convert_video_to_gif(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    output_path = (args.output or input_path.with_suffix(".gif")).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input video does not exist: {input_path}")
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0.")
    if args.width < 0:
        raise SystemExit("--width must be 0 or greater.")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists, pass --overwrite to replace it: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = build_filter(args.fps, args.width)

    with tempfile.TemporaryDirectory(prefix="video_to_gif_") as tmp_dir:
        palette_path = Path(tmp_dir) / "palette.png"

        palette_command = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *build_base_input_args(args),
            "-vf",
            f"{filter_chain},palettegen",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(palette_path),
        ]
        gif_command = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if args.overwrite else "-n",
            *build_base_input_args(args),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{filter_chain} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
            str(output_path),
        ]

        run_ffmpeg(palette_command)
        run_ffmpeg(gif_command)

    return output_path


def main() -> int:
    args = parse_args()
    require_ffmpeg(args.ffmpeg)
    output_path = convert_video_to_gif(args)
    print(f"GIF saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
