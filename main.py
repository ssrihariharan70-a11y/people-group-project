#!/usr/bin/env python3
"""People Group CLI.

Commands:
    demo             Run the scripted crowd-growth demo (no camera needed)
    run              Process a video file or camera with YOLO
    rules init       Create an empty rules file
    rules list       Show stored rules
    rules add        Add a rule  (--rule region:warning:critical[:min_group_size])
    rules remove     Remove a rule by region
    test-connection  Probe the platform endpoint for crowd alert delivery
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from people_group import (
    CrowdAlertConnector,
    Event,
    GroupRule,
    HttpTransport,
    InMemoryTransport,
    PeopleGroupPipeline,
    Region,
    RuleStore,
    ScriptedDetector,
    __version__,
    person_at,
)

DEFAULT_RULES_PATH = "rules.json"


def build_pipeline(args: argparse.Namespace, scripted: bool = False) -> tuple[PeopleGroupPipeline, InMemoryTransport]:
    """Build a pipeline with rules and (optionally) a platform connector."""
    rules = RuleStore(getattr(args, "rules", DEFAULT_RULES_PATH))
    if len(rules) == 0:
        rules.add(GroupRule(region="plaza", warning_at=3, critical_at=5), overwrite=True)

    if getattr(args, "endpoint", None):
        connector = CrowdAlertConnector(
            args.endpoint,
            api_key=getattr(args, "api_key", None),
            transport=HttpTransport(),
        )
        recorder = InMemoryTransport()
    else:
        # Demo mode: record alerts in memory instead of posting to a platform.
        connector = CrowdAlertConnector(
            "in-memory://demo",
            transport=(recorder := InMemoryTransport()),
        )

    pipeline = PeopleGroupPipeline(ScriptedDetector() if scripted else None, rules=rules, connector=connector)
    return pipeline, recorder


def build_demo(pipeline: PeopleGroupPipeline) -> PeopleGroupPipeline:
    """Script a small story: one person, a group forms, crowd grows, crowd clears."""
    pipeline.add_region(Region.rectangle("plaza", 100, 100, 500, 420))
    detector = pipeline.detector  # scripted detector for the demo

    alone = [person_at(180, 250)]
    small_group = [
        person_at(180, 250),
        person_at(230, 255),
        person_at(280, 245),
    ]
    crowd = small_group + [
        person_at(160, 300),
        person_at(320, 300),
        person_at(260, 330),
    ]

    detector.add_scene(0, alone)
    detector.add_scene(1, small_group)
    detector.add_scene(2, small_group)
    detector.add_scene(3, crowd)
    detector.add_scene(4, crowd)
    detector.add_scene(5, [])
    return pipeline


def cmd_demo(args: argparse.Namespace) -> int:
    pipeline, recorder = build_pipeline(args, scripted=True)
    build_demo(pipeline)
    pipeline.subscribe(lambda event: print(f"  event: {event.type.value:<18} {event.message}"))

    print(f"People Group demo v{__version__} | rules: {[r.to_dict() for r in pipeline.rules.all()]}")
    for frame_id in range(6):
        print(f"\n--- frame {frame_id} ---")
        result = pipeline.process_frame(frame_id)
        print(result.display)

    alerts = [e for e in pipeline.events.of_type("crowd_alert")]
    print(f"\nAlerts sent: {len(recorder.requests)} request(s) to {pipeline.connector.endpoint}")
    print(f"Events recorded: {len(pipeline.events)}")
    if recorder.requests:
        print(f"First alert payload: {recorder.requests[0].json()['event']['message']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError:
        print("error: 'run' needs opencv-python (pip install opencv-python)", file=sys.stderr)
        return 2

    from people_group.visualize import annotate  # noqa: PLC0415 - optional (needs cv2)

    pipeline, _ = build_pipeline(args)
    try:
        from people_group import YoloPeopleDetector  # noqa: PLC0415

        pipeline.detector = YoloPeopleDetector(model_name=args.model, confidence=args.conf)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    source = int(args.video) if args.video.isdigit() else args.video
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: cannot open video source {args.video!r}", file=sys.stderr)
        return 2

    # Track the full frame so the region fits any video resolution.
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    region = Region.rectangle("plaza", 0, 0, width, height)
    pipeline.add_region(region)
    pipeline.subscribe(lambda event: print(f"  event: {event.type.value:<18} {event.message}"))

    view = not args.no_view
    writer = None
    frame_id = 0
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, frame_id=frame_id)
            if frame_id % args.display_every == 0:
                print(result.display)
            if view or args.save_video:
                annotated = annotate(frame, result, str(frame_id), region)
                if args.save_video:
                    if writer is None:
                        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                        writer = cv2.VideoWriter(
                            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                    writer.write(annotated)
                if view:
                    cv2.imshow("People Group - detected", annotated)
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
            if view and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()
    print(f"Processed {frame_id} frame(s).")
    if args.save_video:
        print(f"Annotated video written to: {args.save_video}")
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    store = RuleStore(args.rules)
    action = args.rules_action

    if action == "init":
        store.save()
        print(f"Created rules file: {Path(args.rules).resolve()}")
        return 0
    if action == "list":
        if len(store) == 0:
            print("(no rules stored)")
        for rule in store.all():
            print(
                f"{rule.region:<18} warning_at={rule.warning_at} critical_at={rule.critical_at} "
                f"min_group_size={rule.min_group_size} enabled={rule.enabled}"
            )
        return 0
    if action == "add":
        try:
            parts = args.rule.split(":")
            if len(parts) not in (3, 4):
                raise ValueError("expected region:warning:critical[:min_group_size]")
            region, warning, critical = parts[0], int(parts[1]), int(parts[2])
            min_size = int(parts[3]) if len(parts) == 4 else 2
            store.add(GroupRule(region=region, warning_at=warning, critical_at=critical, min_group_size=min_size))
        except (ValueError, KeyError) as exc:
            print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return 2
        print(f"Added rule for '{region}' -> {store.get(region).to_dict()}")
        return 0
    if action == "remove":
        try:
            removed = store.remove(args.region)
        except KeyError as exc:
            print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return 2
        print(f"Removed rule: {removed.to_dict()}")
        return 0
    print(f"error: unknown rules action {action!r}", file=sys.stderr)
    return 2


def cmd_test_connection(args: argparse.Namespace) -> int:
    connector = CrowdAlertConnector(args.endpoint, api_key=args.api_key, transport=HttpTransport())
    ok = connector.test_connection()
    print(f"{'OK' if ok else 'FAILED'}: probe to {connector.endpoint}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="people-group", description="People Group — crowd monitoring pipeline")
    parser.add_argument("--version", action="version", version=f"people-group {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the scripted crowd demo (no camera needed)")
    demo.add_argument("--rules", default=DEFAULT_RULES_PATH, help="rules JSON path")
    demo.add_argument("--endpoint", help="platform endpoint for crowd alerts")
    demo.add_argument("--api-key", dest="api_key", help="platform API key")
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="process a video file or camera index with YOLO")
    run.add_argument("--video", required=True, help="video path or camera index (e.g. 0)")
    run.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    run.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    run.add_argument("--rules", default=DEFAULT_RULES_PATH, help="rules JSON path")
    run.add_argument("--endpoint", help="platform endpoint for crowd alerts")
    run.add_argument("--api-key", dest="api_key", help="platform API key")
    run.add_argument("--max-frames", dest="max_frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    run.add_argument("--display-every", dest="display_every", type=int, default=1, help="print display every N frames")
    run.add_argument("--no-view", dest="no_view", action="store_true", help="disable the live annotated window")
    run.add_argument("--save-video", dest="save_video", metavar="PATH", help="write annotated frames to this .mp4")
    run.set_defaults(func=cmd_run)

    rules = sub.add_parser("rules", help="manage stored group rules")
    rules_sub = rules.add_subparsers(dest="rules_action", required=True)
    for name, help_text in [
        ("init", "create an empty rules file"),
        ("list", "list stored rules"),
        ("add", "add a rule"),
        ("remove", "remove a rule by region"),
    ]:
        leaf = rules_sub.add_parser(name, help=help_text)
        leaf.add_argument("--rules", default=DEFAULT_RULES_PATH, help="rules JSON path")
        leaf.set_defaults(rules_action=name)
    rules_sub.choices["add"].add_argument("--rule", required=True, help="region:warning:critical[:min_group_size]")
    rules_sub.choices["remove"].add_argument("--region", required=True)
    rules.set_defaults(func=cmd_rules)

    probe = sub.add_parser("test-connection", help="probe the platform alert endpoint")
    probe.add_argument("--endpoint", required=True)
    probe.add_argument("--api-key", dest="api_key")
    probe.set_defaults(func=cmd_test_connection)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
