"""
Drawing helpers. Pure rendering -- these never compute anything, so a
mistake here can only ever be cosmetic.
"""

import cv2

from . import config


def draw_2d_box(image, detection, color=None, label=None, thickness=None):
    """Draw one detection's box, with its label above it.

    The label is placed inside the frame when the box is near the top edge,
    otherwise it would be drawn off-screen and silently lost.
    """
    color = color or config.BOX_COLOR
    thickness = thickness or config.BOX_THICKNESS
    x1, y1, x2, y2 = (int(round(v)) for v in (detection.x1, detection.y1, detection.x2, detection.y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label is None:
        label = f"{detection.class_name} {detection.confidence:.2f}"

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
    )
    text_y = y1 - 4 if y1 - text_h - 6 >= 0 else y2 + text_h + 4
    cv2.rectangle(
        image,
        (x1, text_y - text_h - baseline + 2),
        (x1 + text_w + 4, text_y + baseline - 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 2, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        config.TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return image


def draw_caption(image, lines, origin=(10, 10)):
    """Small top-left status block, for frame index / counts / scene name."""
    x, y = origin
    for i, line in enumerate(lines):
        (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = y + i * (th + 10)
        cv2.rectangle(image, (x, top), (x + tw + 8, top + th + base + 4), (0, 0, 0), -1)
        cv2.putText(
            image,
            line,
            (x + 4, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image
