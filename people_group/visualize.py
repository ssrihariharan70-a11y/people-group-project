"""Visual annotation helpers (optional — requires opencv-python).

Draws YOLO detections, groups, the watched region and the count display onto
a video frame. Used by ``main.py run`` (live view) and ``preview_frames.py``.
"""

from __future__ import annotations

import cv2

LEVEL_COLORS = {
    "ok": (60, 180, 75),       # green
    "warning": (0, 165, 255),  # orange
    "critical": (0, 0, 230),   # red
}
GROUP_PALETTE = [
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 128, 255),    # orange
    (180, 255, 0),    # teal
    (128, 128, 255),  # pink
]
ALONE_COLOR = (200, 200, 200)  # light gray
REGION_COLOR = (255, 120, 0)   # blue


def annotate(frame, result, frame_id: str, region):
    """Draw boxes, groups, region and status info onto a copy of the frame."""
    img = frame.copy()
    height, width = img.shape[:2]

    # Region border.
    pts = [tuple(map(int, p)) for p in region.polygon]
    for i in range(len(pts)):
        cv2.line(img, pts[i], pts[(i + 1) % len(pts)], REGION_COLOR, 2)
    cv2.putText(img, region.name, (pts[0][0] + 6, pts[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, REGION_COLOR, 2, cv2.LINE_AA)

    # Member boxes (gray when alone, group colour when grouped) + group boxes.
    for group in result.groups:
        color = GROUP_PALETTE[group.id % len(GROUP_PALETTE)] if group.size >= 2 else ALONE_COLOR
        for det in group.detections:
            b = det.bbox
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 1)
        if group.size >= 2:
            gb = group.bbox
            cv2.rectangle(img, (int(gb.x1), int(gb.y1)), (int(gb.x2), int(gb.y2)), color, 3)
            cv2.putText(img, f"G{group.id} ({group.size})", (int(gb.x1), int(gb.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Status bar (top).
    status = result.statuses[0] if result.statuses else None
    level = result.worst_level.value
    people = status.people_count if status else 0
    groups = status.group_count if status else 0
    largest = max((g.size for g in result.groups if g.size >= 2), default=0)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"frame {frame_id}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{region.name}: people={people} groups={groups} largest={largest}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    badge = level.upper()
    badge_color = LEVEL_COLORS[level]
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = width - tw - 24
    cv2.rectangle(img, (x - 10, 10), (width - 10, 10 + th + 12), badge_color, -1)
    cv2.putText(img, badge, (x, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # ASCII count display (bottom-left) — the same table the CLI prints.
    lines = [ln for ln in result.display.splitlines() if ln.strip()]
    scale, thickness, pad = 0.45, 1, 6
    sizes = [cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for ln in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = len(lines) * 18 + 2 * pad
    y0 = height - box_h - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (8, y0), (8 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (ln, (tw, _)) in enumerate(zip(lines, sizes)):
        color = LEVEL_COLORS.get(level, (255, 255, 255)) if "TOTAL" in ln else (255, 255, 255)
        cv2.putText(img, ln, (8 + pad, y0 + pad + 14 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img
