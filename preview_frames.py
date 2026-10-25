#!/usr/bin/env python3
"""Preview tool: run YOLO on a video and save annotated frames.

For every threshold-level transition (OK -> WARNING -> CRITICAL ...) a frame is
saved to ``test_frames/`` showing the detections, groups, region and count
display (see ``people_group/visualize.py`` for the drawing code).

Usage:
    python preview_frames.py [video] [--out DIR] [--max-frames N] [--conf 0.4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from people_group import PeopleGroupPipeline, Region, RuleStore, YoloPeopleDetector
from people_group.visualize import annotate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Save annotated frames from a video")
    parser.add_argument("video", nargs="?", default="vtest.avi")
    parser.add_argument("--out", default="test_frames")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args(argv)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"error: cannot open {args.video!r}")
        return 2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    region = Region.rectangle("plaza", 0, 0, width, height)

    pipeline = PeopleGroupPipeline(
        YoloPeopleDetector("yolov8n.pt", confidence=args.conf),
        rules=RuleStore("rules.json"),
    )
    pipeline.add_region(region)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    saved, previous_level = [], None
    for frame_id in range(args.max_frames):
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process_frame(frame, frame_id)
        level = result.worst_level.value
        if level != previous_level and len(saved) < args.max_shots:
            shot = annotate(frame, result, str(frame_id), region)
            path = out_dir / f"frame_{frame_id:04d}_{level}.png"
            cv2.imwrite(str(path), shot)
            saved.append(path)
            print(f"saved {path}  (level: {level})")
            previous_level = level
        if len(saved) >= args.max_shots:
            break

    capture.release()
    print(f"{len(saved)} preview frame(s) written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
