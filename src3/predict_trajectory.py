"""
Where will each box be in the next N frames?

Builds on src3.detect_3d: every frame is turned into 3D boxes by tight-fit
geometry, those boxes are followed from frame to frame, and each one's
measured velocity is extended forward to say where it will be.

Two things this gets right that a naive version would not:

  * Tracking happens in the GLOBAL frame, not the camera's. The camera rides
    on a moving car -- ego travels ~4.5 m between nuScenes keyframes -- so in
    camera coordinates a parked car appears to slide backwards, and every
    parked vehicle would be predicted to reverse down the road.

  * The prediction is then drawn relative to where the EGO will be at that
    future moment, not where it is now. What matters for a collision is the
    gap between the two vehicles when they arrive, and that gap closes
    because of both cars' motion.

Output is a camera view with the future boxes drawn fading into the distance,
beside a bird's-eye panel where the trajectories are actually legible --
overlapping wireframes in the image cannot show a path, but a top-down plot
can.

Usage (from the project root):
    python -m src3.predict_trajectory --start 40 --count 8
    python -m src3.predict_trajectory --start 40 --count 8 --horizon 4
"""

import argparse

import cv2
import numpy as np

from src2 import config
from src2.detection import VehicleDetector
from src2.nuscenes_source import NuScenesSource
from src2.tracking import Tracker
from src3.deep3dbox import fit_box_search_yaw, object_corners, project_box, rotation_y

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

BEV_RANGE_M = 60.0          # how far ahead the top-down panel shows
BEV_HALF_WIDTH_M = 25.0
BEV_SIZE = (900, 520)       # width, height in pixels


def detect_boxes(frame, detector):
    """3D boxes for one frame, camera coordinates."""
    K = frame.intrinsics
    out = []
    for det in detector.detect(frame.image):
        box_2d = (det.x1, det.y1, det.x2, det.y2)
        translation, yaw, error = fit_box_search_yaw(box_2d, config.CAR_SIZE_WLH, K)
        if translation is None:
            continue
        if error > 0.5 * ((det.x2 - det.x1) + (det.y2 - det.y1)):
            continue        # the fit failed; drawing it would look confident and be wrong
        out.append({"translation": translation, "yaw": yaw, "det": det})
    return out


def draw_wireframe(image, uv, colour, thickness=2):
    pts = uv.T.astype(int)
    for a, b in EDGES:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), colour, thickness, cv2.LINE_AA)


def draw_footprint(image, uv, colour, thickness=1):
    """Only the box's base rectangle -- where it meets the road.

    Future positions are drawn this way rather than as full wireframes. A
    full box is twelve lines; with several vehicles and several horizons the
    image turns into a thicket and none of it can be read. The footprint is
    four lines, sits on the road where a driver looks, and still shows
    position, size and heading.
    """
    pts = uv.T.astype(int)[[0, 1, 2, 3]]
    cv2.polylines(image, [pts], True, colour, thickness, cv2.LINE_AA)


def base_centre_pixel(uv):
    """Pixel at the middle of the box's base, for drawing a motion trail."""
    pts = uv.T[[0, 1, 2, 3]]
    return tuple(int(v) for v in pts.mean(axis=0))


def to_bev_pixel(x_m, z_m):
    """Ground-plane metres -> pixel in the top-down panel.

    Ego sits at the bottom centre looking up the image, which is how a driver
    reads a map.
    """
    w, h = BEV_SIZE
    px = w / 2 + (x_m / BEV_HALF_WIDTH_M) * (w / 2)
    py = h - (z_m / BEV_RANGE_M) * h
    return int(px), int(py)


def draw_bev_background():
    w, h = BEV_SIZE
    panel = np.full((h, w, 3), 24, np.uint8)
    for z in range(10, int(BEV_RANGE_M) + 1, 10):
        _, py = to_bev_pixel(0, z)
        cv2.line(panel, (0, py), (w, py), (55, 55, 55), 1)
        cv2.putText(panel, f"{z}m", (6, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (110, 110, 110), 1, cv2.LINE_AA)
    for x in (-20, -10, 0, 10, 20):
        px, _ = to_bev_pixel(x, 0)
        cv2.line(panel, (px, 0), (px, h), (55, 55, 55), 1)
    # ego marker
    ex, ey = to_bev_pixel(0, 0)
    cv2.drawMarker(panel, (ex, ey - 8), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
    cv2.putText(panel, "ego", (ex - 14, ey - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def bev_vehicle(panel, x_m, z_m, yaw, colour, thickness=2):
    """A vehicle's footprint, seen from above."""
    w, l, _ = config.CAR_SIZE_WLH
    corners = np.array([[-w / 2, w / 2, w / 2, -w / 2], [l / 2, l / 2, -l / 2, -l / 2]])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rotated = np.array([[c, -s], [s, c]]) @ corners
    pts = [to_bev_pixel(x_m + rotated[0, i], z_m + rotated[1, i]) for i in range(4)]
    cv2.polylines(panel, [np.array(pts)], True, colour, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3,
                        help="how many frames ahead to predict (0.5 s each)")
    parser.add_argument("--bev", action="store_true",
                        help="also append the top-down panel underneath")
    parser.add_argument("--footprints", action="store_true",
                        help="draw future positions as ground footprints instead of "
                             "full boxes -- much less cluttered in busy scenes")
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector()
    tracker = Tracker()
    out_dir = config.PROJECT_ROOT / "output3"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_dt = 0.5           # nuScenes keyframe spacing
    scene = None
    previous_global_ego = None

    for index in range(args.start, min(args.start + args.count, len(source))):
        frame = source.frame(index)
        if frame.scene_name != scene:
            scene = frame.scene_name
            tracker = Tracker()
            previous_global_ego = None

        boxes = detect_boxes(frame, detector)
        canvas = frame.image.copy()
        panel = draw_bev_background()

        # Ego's own velocity, from where its camera was last frame. Needed to
        # place predictions in the frame the ego will actually occupy.
        ego_now = frame.cam_to_global_t
        ego_velocity = np.zeros(3)
        if previous_global_ego is not None:
            ego_velocity = (ego_now - previous_global_ego) / frame_dt
        previous_global_ego = ego_now.copy()

        if boxes:
            centres_cam = np.stack([b["translation"] for b in boxes], axis=1)
            centres_global = frame.to_global(centres_cam)
            ids = tracker.update(
                [centres_global[:2, i] for i in range(centres_global.shape[1])],
                frame.timestamp_s,
            )
        else:
            ids = []

        predicted_count = 0
        for box, track_id in zip(boxes, ids):
            track = tracker.tracks[track_id]
            distance = float(box["translation"][2])
            colour = ((0, 90, 255) if distance < 15
                      else (0, 200, 255) if distance < 30 else (0, 220, 0))

            uv = project_box(box["translation"], config.CAR_SIZE_WLH, box["yaw"], frame.intrinsics)
            if uv is None or not np.isfinite(uv).all():
                continue
            draw_wireframe(canvas, uv, colour)
            bev_vehicle(panel, float(box["translation"][0]), distance, box["yaw"], colour, 2)

            velocity = track.velocity()
            speed = float(np.hypot(velocity[0], velocity[1])) if velocity is not None else 0.0
            label = f"#{track_id} {distance:.0f}m"
            if velocity is not None:
                label += f" {speed:.0f}m/s"
            cv2.putText(canvas, label,
                        (int(box["det"].x1), max(int(box["det"].y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)

            if velocity is None:
                continue
            predicted_count += 1

            here = centres_global[:, ids.index(track_id)]
            trail_image = [base_centre_pixel(uv)]
            trail_bev = []
            for step in range(1, args.horizon + 1):
                ahead = step * frame_dt
                # The vehicle's future position in the world...
                future_global = np.array([here[0] + velocity[0] * ahead,
                                          here[1] + velocity[1] * ahead,
                                          here[2]])
                # ...seen from where the ego will be by then. Both vehicles
                # move, and it is the gap between them that decides a collision.
                future_ego = ego_now + ego_velocity * ahead
                relative = frame.cam_to_global_R.T @ (future_global - future_ego)

                fade = 0.8 ** step
                faded = tuple(int(c * fade) for c in colour)
                bev_vehicle(panel, float(relative[0]), float(relative[2]),
                            box["yaw"], faded, 1)
                trail_bev.append(to_bev_pixel(float(relative[0]), float(relative[2])))

                future_uv = project_box(relative, config.CAR_SIZE_WLH,
                                        box["yaw"], frame.intrinsics)
                if future_uv is None or not np.isfinite(future_uv).all():
                    continue
                # Full wireframe, so a predicted position reads as the same
                # kind of object as the current one rather than as a mark on
                # the road. Thin for intermediate steps and thicker for the
                # last, which is the one the label belongs to. In a busy scene
                # this stacks up; --footprints trades the 3D look for clarity.
                if args.footprints:
                    draw_footprint(canvas, future_uv, faded,
                                   2 if step == args.horizon else 1)
                else:
                    draw_wireframe(canvas, future_uv, faded,
                                   2 if step == args.horizon else 1)
                trail_image.append(base_centre_pixel(future_uv))
                if step == args.horizon:
                    px, py = trail_image[-1]
                    cv2.putText(canvas, f"+{ahead:.1f}s", (px + 4, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, faded, 1, cv2.LINE_AA)

            # The trail along the road makes the direction of travel obvious
            # even where the footprints overlap.
            if len(trail_image) > 1:
                cv2.polylines(canvas, [np.array(trail_image)], False, colour, 1, cv2.LINE_AA)
                for px, py in trail_image[1:]:
                    cv2.circle(canvas, (px, py), 2, colour, -1)
            if len(trail_bev) > 1:
                cv2.polylines(panel, [np.array(trail_bev)], False, colour, 1, cv2.LINE_AA)

        cv2.putText(panel, f"predicted {args.horizon} frames ahead "
                           f"({args.horizon * frame_dt:.1f}s), faded = further ahead",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {index}  {frame.scene_name}   camera only",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(boxes)} boxes, {predicted_count} predicted "
                            f"{args.horizon * frame_dt:.1f}s ahead  "
                            f"(bright = now, faded = future)",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        output = canvas
        if args.bev:
            panel_resized = cv2.resize(panel, (canvas.shape[1], int(
                BEV_SIZE[1] * canvas.shape[1] / BEV_SIZE[0])))
            output = np.vstack([canvas, panel_resized])
        path = out_dir / f"traj_{index:04d}.jpg"
        cv2.imwrite(str(path), output)
        print(f"  frame {index:3d}  {len(boxes)} boxes, {predicted_count} predicted -> {path.name}")


if __name__ == "__main__":
    main()
