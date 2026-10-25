# People in Group Project

YOLO people detection with **group-in-region** logic, group-count thresholds,
event rules, and crowd alerts to a platform. Implements the five project tasks:

| # | Task | Where it lives |
|---|------|----------------|
| 1 | YOLO people detection and group-in-region logic | `people_group/detection.py`, `people_group/grouping.py` |
| 2 | Group count display and thresholds | `people_group/thresholds.py` |
| 3 | Group events and rule storage | `people_group/events.py`, `people_group/rules.py` |
| 4 | Connect crowd alerts to platform | `people_group/alerts.py` |
| 5 | Group threshold tests and sign-off | `tests/`, sign-off checklist below |

The core pipeline is pure standard-library Python. `ultralytics` + `opencv-python`
are optional extras, only needed for real camera/video YOLO inference — the demo
and the whole test suite run without them.

## Install

```bash
pip install -r requirements.txt      # pytest for the test suite
pip install -e ".[yolo]"             # optional: real YOLO inference
```

## Quickstart — scripted demo (no camera needed)

```bash
python main.py demo
```

A six-frame story plays out in a `plaza` region: one person strolls in, a group
of three forms (WARNING at 3), the crowd grows to six (CRITICAL at 5, crowd
alert recorded), then disperses (THRESHOLD_CLEARED). Each frame prints the
count display and every event as it happens.

Send alerts to a real platform endpoint instead of the in-memory recorder:

```bash
python main.py demo --endpoint https://platform.example.com/api/crowd --api-key YOUR_KEY
```

## Real video / camera with YOLO

```bash
python main.py run --video crowd.mp4 --rules rules.json --endpoint https://platform.example.com/api/crowd
python main.py run --video 0          # webcam index 0
python main.py run --video crowd.mp4 --model yolov8n.pt --conf 0.4 --max-frames 500
```

A live window (`People Group - detected`) shows each frame with YOLO boxes,
group outlines, the region border and the count overlay. Press `q` to quit.
Use `--no-view` to run headless, or `--save-video out.mp4` to write the
annotated video to a file.

## Rule management (task 3)

Rules live in a JSON file (`rules.json` by default) and survive restarts:

```bash
python main.py rules init --rules rules.json
python main.py rules add --rules rules.json --rule plaza:3:5:2   # region:warning:critical[:min_group_size]
python main.py rules list --rules rules.json
python main.py rules remove --rules rules.json --region plaza
```

Example `rules.json`:

```json
{
  "version": 1,
  "rules": [
    {"region": "plaza", "min_group_size": 2, "warning_at": 3, "critical_at": 5, "enabled": true}
  ]
}
```

A rule's `region` should match a region registered in code
(`pipeline.add_region(Region.rectangle("plaza", ...))`). A rule whose region has
no registered geometry acts as a **global rule** and counts every group.

## Platform connectivity (task 4)

```bash
python main.py test-connection --endpoint https://platform.example.com/api/crowd --api-key YOUR_KEY
```

The connector POSTs JSON (`kind: crowd_alert`, full event body) with an
`X-Api-Key` header. 2xx counts as delivered; 4xx fails immediately (bad
request); 5xx and network errors are retried with exponential backoff. The
transport is pluggable — tests and the demo use `InMemoryTransport`.

## How grouping works

- Detections are clustered by **single-linkage** on footprint distance
  (bottom-center of each box): people within `max_link_distance` (default 90 px,
  chained) form one group.
- A group is "in" a region when at least 50 % (`region_vote_threshold`) of its
  members' footprints fall inside the region polygon (ray-casting test;
  rectangles via `Region.rectangle`).
- Thresholds escalate immediately but de-escalate with **hysteresis** (default 1
  person), so a count hovering at a boundary does not flap between levels.

## Using it as a library

```python
from people_group import (
    CrowdAlertConnector, GroupRule, PeopleGroupPipeline, Region, RuleStore, YoloPeopleDetector,
)

pipeline = PeopleGroupPipeline(
    YoloPeopleDetector("yolov8n.pt", confidence=0.4),
    rules=RuleStore("rules.json"),
    connector=CrowdAlertConnector("https://platform.example.com/api/crowd", api_key="KEY"),
)
pipeline.add_region(Region.rectangle("plaza", 100, 100, 500, 420))
pipeline.rules.add(GroupRule(region="plaza", warning_at=3, critical_at=5))
pipeline.subscribe(lambda event: print(event.type.value, event.message))

for frame_id, frame in enumerate(video_frames):
    result = pipeline.process_frame(frame, frame_id)
    if result.worst_level.value == "critical":
        ...  # result.display shows the full count table
```

## Tests and sign-off (task 5)

```bash
python -m pytest
```

| Task | Sign-off criterion | Covered by |
|------|--------------------|------------|
| 1. YOLO detection + group-in-region | Nearby people chain into groups; far people stay separate; region binding by majority vote | `tests/test_grouping.py`, `tests/test_detection.py` |
| 2. Count display and thresholds | Boundary counts map to OK/WARNING/CRITICAL; hysteresis prevents flapping; display renders all rows | `tests/test_thresholds.py` |
| 3. Events and rule storage | Events record + dispatch; rules round-trip through JSON and persist across store instances | `tests/test_events.py`, `tests/test_rules.py` |
| 4. Crowd alerts to platform | Payload/headers correct; 5xx retried then given up; 4xx not retried; connectivity probe works | `tests/test_alerts.py` |
| 5. End-to-end pipeline | Frame-by-frame story emits formed/warning/critical/alert/dissolved/cleared in order; alert hits the transport | `tests/test_pipeline.py` |

**Sign-off:** all tasks are considered done when `python -m pytest` passes with
the suite above green.
