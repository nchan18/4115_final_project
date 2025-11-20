#!/usr/bin/env python3
"""
Extract frames from all videos in the `videos/` folder and save N frames per second
into the `pictures/` folder under per-video subfolders.

Defaults: 2 frames per second.

Usage:
    python extract_frames.py --videos-dir videos --output-dir pictures --fps 2

Requires: opencv-python
"""
import argparse
from pathlib import Path
import cv2
import math
import sys

COMMON_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.MP4', '.AVI', '.MOV'}


def extract_from_video(video_path: Path, out_dir: Path, target_fps: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = None
    if video_fps and video_fps > 0:
        duration_sec = frame_count / video_fps if frame_count > 0 else None

    sample_period = 1.0 / float(target_fps)

    saved = 0
    if duration_sec and duration_sec > 0:
        # Seek by time (ms) to get consistent sampling independent of source fps.
        t = 0.0
        idx = 0
        while t < duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                # sometimes last few seeks fail; break if no frame
                break
            out_name = out_dir / f"{video_path.stem}_frame_{idx:06d}.jpg"
            # write with reasonable JPEG quality
            cv2.imwrite(str(out_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
            idx += 1
            t += sample_period
    else:
        # Fallback: read sequentially and sample by frame count if fps unknown
        if not video_fps or video_fps <= 0:
            video_fps = 30.0
        interval = max(1, int(round(video_fps / target_fps)))
        idx = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_name = out_dir / f"{video_path.stem}_frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                saved += 1
                idx += 1
            frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames from {video_path.name} -> {out_dir}")
    return saved


def find_videos(videos_dir: Path):
    if not videos_dir.exists():
        return []
    vids = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and (p.suffix in COMMON_EXTS):
            vids.append(p)
        else:
            # try opening non-standard extensions too
            if p.is_file():
                vids.append(p)
    return vids


def main(argv=None):
    parser = argparse.ArgumentParser(prog="extract_frames")
    parser.add_argument('--videos-dir', type=Path, default=Path('videos'), help='Directory containing videos')
    parser.add_argument('--output-dir', type=Path, default=Path('pictures2'), help='Directory to save frames')
    parser.add_argument('--fps', type=float, default=1/4.0, help='Frames per second to save')
    parser.add_argument('--per-video-subdir', action='store_true', help='Save frames into per-video subdirectories (default: True)')
    args = parser.parse_args(argv)

    videos_dir = args.videos_dir
    out_root = args.output_dir
    target_fps = float(args.fps)

    if not videos_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    vids = find_videos(videos_dir)
    if not vids:
        print(f"No videos found in {videos_dir}")
        sys.exit(0)

    total_saved = 0
    for v in vids:
        # create per-video folder
        if args.per_video_subdir:
            out_dir = out_root / v.stem
        else:
            out_dir = out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            saved = extract_from_video(v, out_dir, target_fps)
            total_saved += saved
        except Exception as e:
            print(f"Error processing {v}: {e}")

    print(f"Done. Total frames saved: {total_saved}")


if __name__ == '__main__':
    main()
