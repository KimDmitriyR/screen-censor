from flask import Flask, request, jsonify
import cv2
import json
import math
import os
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from bodyparts_recipe import MODEL_PART_TO_SERVER_PART, SERVER_SETTING_TO_MODEL_PART

app = Flask(__name__)

MODEL_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = MODEL_DIR / "yolo26n-pose.pt"
LEGACY_SEG_MODEL_PATH = MODEL_DIR / "yolo26n-seg.pt"
TRAINED_SEG_MODEL_PATH = MODEL_DIR / "bodyparts-seg-best.pt"
SEG_MODEL_PATH = Path(
    os.getenv(
        "BODY_PARTS_SEG_MODEL",
        str(TRAINED_SEG_MODEL_PATH if TRAINED_SEG_MODEL_PATH.exists() else LEGACY_SEG_MODEL_PATH),
    )
)

pose_model = YOLO(str(POSE_MODEL_PATH))
seg_model = YOLO(str(SEG_MODEL_PATH))

KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

TRACKS = {}
NEXT_PERSON_ID = 1
MAX_TRACK_AGE = 1.2
MATCH_DISTANCE = 220.0
MALE_GROIN_PERSIST_SECONDS = 0.9


def normalized_seg_names(model):
    raw_names = model.names
    items = raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names)
    return {int(idx): str(name).strip().lower().replace(" ", "_") for idx, name in items}


SEG_MODEL_NAMES = normalized_seg_names(seg_model)
SEG_MODEL_IDS_BY_NAME = {name: idx for idx, name in SEG_MODEL_NAMES.items()}

DEFAULT_SEG_MIN_CONF = 0.25
SEG_CLASS_MIN_CONF = {
    "silhouette": 0.02,
    "shoulders": 0.40,
    "wrists": 0.40,
    "chest": 0.40,
    "back": 0.45,
    "armpits": 0.45,
    "navel": 0.45,
    "hips": 0.40,
    "buttocks": 0.50,
    "male_groin": 0.55,
    "intimate_front": 0.55,
}


def clamp(v, lo, hi):
    return max(lo, min(int(round(v)), hi))


def seg_min_conf_for_part(part_name):
    return SEG_CLASS_MIN_CONF.get(part_name, DEFAULT_SEG_MIN_CONF)


def update_seg_support(seg_support, part_name, conf):
    if not part_name:
        return
    seg_support[part_name] = max(float(conf), float(seg_support.get(part_name, 0.0)))


def seg_support_conf(seg_support, part_name):
    return float(seg_support.get(part_name, 0.0))


def parse_settings():
    raw = request.form.get("settings", "{}")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def point_xy(kpts, idx, w, h):
    # Ultralytics keypoints.xy already are in pixel coordinates.
    x, y = kpts[idx]
    return clamp(x, 0, w - 1), clamp(y, 0, h - 1)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def toward(p1, p2, factor):
    return (p1[0] + (p2[0] - p1[0]) * factor, p1[1] + (p2[1] - p1[1]) * factor)


def extend_segment(a, b, extension=0.45):
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    return (bx + dx * extension, by + dy * extension)


def ellipse_polygon(cx, cy, rx, ry, n=32):
    pts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append({
            "x": int(round(cx + rx * math.cos(a))),
            "y": int(round(cy + ry * math.sin(a)))
        })
    return pts


def circle_polygon(cx, cy, r, n=20):
    return ellipse_polygon(cx, cy, r, r, n=n)


def capsule_polygon(p1, p2, radius=28, n=14):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length < 1:
        return circle_polygon(x1, y1, radius, n=n * 2)

    ang = math.atan2(dy, dx)
    pts = []

    for i in range(n + 1):
        t = ang - math.pi / 2 + (math.pi * i / n)
        pts.append({
            "x": int(round(x2 + radius * math.cos(t))),
            "y": int(round(y2 + radius * math.sin(t)))
        })

    for i in range(n + 1):
        t = ang + math.pi / 2 + (math.pi * i / n)
        pts.append({
            "x": int(round(x1 + radius * math.cos(t))),
            "y": int(round(y1 + radius * math.sin(t)))
        })

    return pts


def resample_points(points, target_n=32):
    if points is None:
        return None

    arr = np.array(points, dtype=np.float32)
    if len(arr) < 3:
        return None

    closed = np.vstack([arr, arr[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())

    if total <= 1e-6:
        out = [{"x": int(p[0]), "y": int(p[1])} for p in arr[:target_n]]
        return out if len(out) >= 3 else None

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0, total, target_n, endpoint=False)

    out = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = max(0, min(idx, len(arr) - 1))

        t0 = cum[idx]
        t1 = cum[idx + 1]
        p0 = closed[idx]
        p1 = closed[idx + 1]

        if abs(t1 - t0) < 1e-6:
            p = p0
        else:
            a = (t - t0) / (t1 - t0)
            p = p0 * (1.0 - a) + p1 * a

        out.append({"x": int(round(p[0])), "y": int(round(p[1]))})

    return out if len(out) >= 3 else None


def pose_silhouette_polygon(kpts, confs, w, h):
    pts = []
    for idx in [
        KP["nose"],
        KP["left_eye"], KP["right_eye"],
        KP["left_shoulder"], KP["right_shoulder"],
        KP["left_elbow"], KP["right_elbow"],
        KP["left_wrist"], KP["right_wrist"],
        KP["left_hip"], KP["right_hip"],
        KP["left_knee"], KP["right_knee"],
        KP["left_ankle"], KP["right_ankle"],
    ]:
        if confs is None or confs[idx] >= 0.35:
            pts.append(point_xy(kpts, idx, w, h))

    if len(pts) < 6:
        return None

    arr = np.array(pts, dtype=np.float32)
    center = arr.mean(axis=0, keepdims=True)
    expanded = center + (arr - center) * 1.16
    hull = cv2.convexHull(expanded.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return None

    return [
        {"x": clamp(pt[0], 0, w - 1), "y": clamp(pt[1], 0, h - 1)}
        for pt in hull
    ]


def center_from_kpts(kpts, confs, w, h):
    preferred = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    pts = []

    def kp_ok(idx):
        return confs is None or confs[idx] >= 0.25

    for idx in preferred:
        if kp_ok(idx):
            pts.append(point_xy(kpts, idx, w, h))

    if len(pts) < 2:
        for idx in [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]:
            if kp_ok(idx):
                pts.append(point_xy(kpts, idx, w, h))

    if not pts:
        return None

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def detect_active_bounds(frame, diff_threshold=12.0, min_run=8):
    h, w = frame.shape[:2]
    border_samples = np.vstack([
        frame[0, :, :],
        frame[-1, :, :],
        frame[:, 0, :],
        frame[:, -1, :],
    ]).astype(np.float32)
    border_color = np.median(border_samples, axis=0)
    frame_f = frame.astype(np.float32)
    col_diff = np.mean(np.abs(frame_f - border_color), axis=(0, 2))
    row_diff = np.mean(np.abs(frame_f - border_color), axis=(1, 2))

    def find_start(vals):
        for i in range(0, max(1, len(vals) - min_run)):
            if np.mean(vals[i : i + min_run]) > diff_threshold:
                return i
        return 0

    def find_end(vals):
        for i in range(len(vals) - min_run, 0, -1):
            if np.mean(vals[i - min_run : i]) > diff_threshold:
                return i - 1
        return len(vals) - 1

    x0 = find_start(col_diff)
    x1 = find_end(col_diff)
    y0 = find_start(row_diff)
    y1 = find_end(row_diff)

    if x1 <= x0:
        x0, x1 = 0, w - 1
    if y1 <= y0:
        y0, y1 = 0, h - 1

    return (int(x0), int(y0), int(x1), int(y1))


def prune_tracks(now):
    dead = []
    for pid, tr in TRACKS.items():
        if now - tr["last_seen"] > MAX_TRACK_AGE:
            dead.append(pid)
    for pid in dead:
        TRACKS.pop(pid, None)


def assign_person_ids(centers):
    global NEXT_PERSON_ID

    now = time.monotonic()
    prune_tracks(now)

    active = list(TRACKS.items())
    used = set()
    assigned = []

    for center in centers:
        best_pid = None
        best_dist = MATCH_DISTANCE

        for pid, tr in active:
            if pid in used:
                continue

            tx, ty = tr["center"]
            dist = math.hypot(center[0] - tx, center[1] - ty)
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_pid is None:
            best_pid = NEXT_PERSON_ID
            NEXT_PERSON_ID += 1
            track = {
                "center": center,
                "last_seen": now,
                "seen_count": 1,
                "male_groin_persist_until": 0.0,
                "male_groin_last_seen": 0.0,
            }
        else:
            prev = TRACKS.get(best_pid, {})
            track = {
                "center": center,
                "last_seen": now,
                "seen_count": int(prev.get("seen_count", 0)) + 1,
                "male_groin_persist_until": float(prev.get("male_groin_persist_until", 0.0)),
                "male_groin_last_seen": float(prev.get("male_groin_last_seen", 0.0)),
            }

        used.add(best_pid)
        TRACKS[best_pid] = track
        assigned.append(best_pid)

    return assigned


def match_seg_to_pose(seg_centers, pose_centers, pose_ids):
    matched = []
    used = set()

    for c in seg_centers:
        best_pid = None
        best_dist = MATCH_DISTANCE

        for center, pid in zip(pose_centers, pose_ids):
            if pid in used:
                continue
            dist = math.hypot(c[0] - center[0], c[1] - center[1])
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_pid is None:
            global NEXT_PERSON_ID
            best_pid = NEXT_PERSON_ID
            NEXT_PERSON_ID += 1

        used.add(best_pid)
        matched.append(best_pid)

    return matched


def requested_seg_class_ids(settings):
    requested_names = set()

    for setting_key, enabled in settings.items():
        if not enabled:
            continue

        model_part = SERVER_SETTING_TO_MODEL_PART.get(setting_key)
        if model_part in SEG_MODEL_IDS_BY_NAME:
            requested_names.add(model_part)
        elif setting_key == "silhouette" and "person" in SEG_MODEL_IDS_BY_NAME:
            requested_names.add("person")

    class_ids = sorted(SEG_MODEL_IDS_BY_NAME[name] for name in requested_names)
    return class_ids


def seg_class_name_to_server_part(class_name):
    if class_name == "person":
        return "silhouette"
    return MODEL_PART_TO_SERVER_PART.get(class_name, class_name)


def build_person_polygons(
    kpts,
    confs,
    w,
    h,
    pid,
    settings,
    seg_support,
    pose_box_conf=0.0,
    pose_box_xyxy=None,
    active_bounds=None,
    track_state=None,
    now=None,
):
    polys = []

    def kp_conf(idx):
        if confs is None:
            return 1.0
        return float(confs[idx])

    def kp_ok(idx, min_conf=0.25):
        return kp_conf(idx) >= min_conf

    def kp(idx):
        return point_xy(kpts, idx, w, h)

    left_sh = kp(KP["left_shoulder"]) if kp_ok(KP["left_shoulder"], 0.25) else None
    right_sh = kp(KP["right_shoulder"]) if kp_ok(KP["right_shoulder"], 0.25) else None
    left_hip = kp(KP["left_hip"]) if kp_ok(KP["left_hip"], 0.25) else None
    right_hip = kp(KP["right_hip"]) if kp_ok(KP["right_hip"], 0.25) else None

    shoulder_anchor = kp_ok(KP["left_shoulder"], 0.55) and kp_ok(KP["right_shoulder"], 0.55)
    hip_anchor = kp_ok(KP["left_hip"], 0.45) and kp_ok(KP["right_hip"], 0.45)
    torso_anchor = shoulder_anchor and hip_anchor
    lower_body_anchor = torso_anchor and (kp_ok(KP["left_knee"], 0.35) or kp_ok(KP["right_knee"], 0.35))

    silhouette_conf = seg_support_conf(seg_support, "silhouette")
    torso_conf = seg_support_conf(seg_support, "torso")
    face_conf = seg_support_conf(seg_support, "face")
    feet_conf = seg_support_conf(seg_support, "feet")
    upper_body_seg_conf = max(torso_conf, silhouette_conf)
    max_face_pose_conf = max(kp_conf(KP["nose"]), kp_conf(KP["left_eye"]), kp_conf(KP["right_eye"]))
    if active_bounds is None:
        active_x0, active_y0, active_x1, active_y1 = 0, 0, w - 1, h - 1
    else:
        active_x0, active_y0, active_x1, active_y1 = active_bounds
    active_w = max(1.0, float(active_x1 - active_x0 + 1))
    active_h = max(1.0, float(active_y1 - active_y0 + 1))
    side_pad_left = float(active_x0)
    side_pad_right = float(w - 1 - active_x1)
    top_pad = float(active_y0)
    bottom_pad = float(h - 1 - active_y1)
    has_side_padding = side_pad_left >= 40.0 and side_pad_right >= 40.0
    has_top_bottom_padding = top_pad >= 40.0 and bottom_pad >= 40.0
    has_any_padding = has_side_padding or has_top_bottom_padding
    stable_track = track_state is not None and int(track_state.get("seen_count", 0)) >= 2
    male_groin_recent = (
        track_state is not None
        and now is not None
        and float(track_state.get("male_groin_persist_until", 0.0)) >= now
    )

    torso_center = None
    torso_w = 140
    torso_h = 180
    hip_mid = None
    groin_center = None
    groin_inside_active = False
    lower_body_visible = False
    shoulder_y_norm = 0.0
    hip_y_norm = 0.0
    pose_box_top_norm = 0.0

    if left_sh is not None and right_sh is not None and left_hip is not None and right_hip is not None:
        sh_mid = midpoint(left_sh, right_sh)
        hip_mid = midpoint(left_hip, right_hip)

        # Narrower torso polygon to avoid swallowing arms.
        top_left = toward(left_sh, sh_mid, 0.28)
        top_right = toward(right_sh, sh_mid, 0.28)
        bot_left = toward(left_hip, hip_mid, 0.14)
        bot_right = toward(right_hip, hip_mid, 0.14)

        xs = [top_left[0], top_right[0], bot_right[0], bot_left[0]]
        ys = [top_left[1], top_right[1], bot_right[1], bot_left[1]]

        torso_center = ((sh_mid[0] + hip_mid[0]) / 2.0, (sh_mid[1] + hip_mid[1]) / 2.0)
        torso_w = max(90, int(abs(top_right[0] - top_left[0]) * 1.08))
        torso_h = max(110, int(abs(hip_mid[1] - sh_mid[1]) * 1.35))
        groin_center = (hip_mid[0], hip_mid[1] + torso_h * 0.18)
        shoulder_y_norm = float((sh_mid[1] - active_y0) / active_h)
        hip_y_norm = float((hip_mid[1] - active_y0) / active_h)
        if pose_box_xyxy is not None and len(pose_box_xyxy) == 4:
            pose_box_top_norm = float((pose_box_xyxy[1] - active_y0) / active_h)
        else:
            pose_box_top_norm = shoulder_y_norm
        groin_inside_active = (
            active_x0 + active_w * 0.10 <= groin_center[0] <= active_x1 - active_w * 0.10
            and active_y0 + active_h * 0.18 <= groin_center[1] <= active_y1 - active_h * 0.06
        )
        knee_visible = kp_ok(KP["left_knee"], 0.35) or kp_ok(KP["right_knee"], 0.35)
        ankle_visible = kp_ok(KP["left_ankle"], 0.30) or kp_ok(KP["right_ankle"], 0.30)
        lower_body_visible = knee_visible and ankle_visible and groin_inside_active

        if settings.get("torso", True):
            polys.append({
                "id": f"person_{pid}_torso",
                "person_id": pid,
                "part": "torso",
                "points": [
                    {"x": int(round(top_left[0])), "y": int(round(top_left[1]))},
                    {"x": int(round(top_right[0])), "y": int(round(top_right[1]))},
                    {"x": int(round(bot_right[0])), "y": int(round(bot_right[1]))},
                    {"x": int(round(bot_left[0])), "y": int(round(bot_left[1]))}
                ]
            })

    # FACE
    if settings.get("face", True):
        face_pts = []
        for idx in [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]:
            if kp_ok(idx):
                face_pts.append(kp(idx))

        if face_pts:
            xs = [p[0] for p in face_pts]
            ys = [p[1] for p in face_pts]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Smaller and more stable than before.
            rx = max(22, (max(xs) - min(xs)) / 2 + 22)
            ry = max(22, (max(ys) - min(ys)) / 2 + 26)

            polys.append({
                "id": f"person_{pid}_face",
                "person_id": pid,
                "part": "face",
                "points": ellipse_polygon(cx, cy, rx, ry, n=30)
            })

    # EYES
    if settings.get("eyes", False):
        for eye_idx, side in [(KP["left_eye"], "l"), (KP["right_eye"], "r")]:
            if kp_ok(eye_idx):
                ex, ey = kp(eye_idx)
                polys.append({
                    "id": f"person_{pid}_eye_{side}",
                    "person_id": pid,
                    "part": "eyes",
                    "points": circle_polygon(ex, ey, 12, n=16)
                })

    # SHOULDERS
    if settings.get("shoulders", False) and torso_anchor and pose_box_conf >= 0.85:
        for sh_idx, side in [(KP["left_shoulder"], "l"), (KP["right_shoulder"], "r")]:
            if kp_ok(sh_idx, 0.60):
                sx, sy = kp(sh_idx)
                polys.append({
                    "id": f"person_{pid}_shoulder_{side}",
                    "person_id": pid,
                    "part": "shoulders",
                    "points": circle_polygon(sx, sy, 18, n=16)
                })

    # TORSO-related extras
    if torso_center is not None:
        cx, cy = torso_center

        if settings.get("chest", False):
            polys.append({
                "id": f"person_{pid}_chest",
                "person_id": pid,
                "part": "chest",
                "points": ellipse_polygon(cx, cy - torso_h * 0.16, torso_w * 0.38, torso_h * 0.20, n=26)
            })

        allow_back_fallback = (
            settings.get("back", False)
            and torso_anchor
            and pose_box_conf >= 0.80
            and silhouette_conf >= 0.20
            and face_conf < 0.25
            and max_face_pose_conf < 0.95
        )

        if allow_back_fallback:
            polys.append({
                "id": f"person_{pid}_back",
                "person_id": pid,
                "part": "back",
                "points": ellipse_polygon(cx, cy + torso_h * 0.03, torso_w * 0.42, torso_h * 0.24, n=26)
            })

        if settings.get("armpits", False) and torso_anchor and pose_box_conf >= 0.80 and upper_body_seg_conf >= 0.55:
            for sh_idx, el_idx, side in [
                (KP["left_shoulder"], KP["left_elbow"], "l"),
                (KP["right_shoulder"], KP["right_elbow"], "r")
            ]:
                if kp_ok(sh_idx, 0.55) and kp_ok(el_idx, 0.55):
                    sh = kp(sh_idx)
                    el = kp(el_idx)
                    lifted = el[1] <= sh[1] + torso_h * 0.25
                    opened = abs(el[0] - sh[0]) >= torso_w * 0.18
                    if not (lifted and opened):
                        continue
                    mid = midpoint(sh, el)
                    pit = toward(mid, torso_center, 0.28)
                    polys.append({
                        "id": f"person_{pid}_armpit_{side}",
                        "person_id": pid,
                        "part": "armpits",
                        "points": circle_polygon(pit[0], pit[1], max(12, int(torso_w * 0.10)), n=16)
                    })

        if settings.get("navel", False):
            polys.append({
                "id": f"person_{pid}_navel",
                "person_id": pid,
                "part": "navel",
                "points": circle_polygon(cx, cy + torso_h * 0.15, max(10, int(torso_w * 0.08)), n=16)
            })

        allow_buttocks_fallback = (
            settings.get("buttocks", False)
            and torso_anchor
            and lower_body_anchor
            and left_hip is not None
            and right_hip is not None
            and pose_box_conf >= 0.90
            and silhouette_conf >= 0.95
            and 0.10 <= torso_conf <= 0.55
            and feet_conf >= 0.40
            and (face_conf <= 0.30 or feet_conf >= 0.70)
        )

        if allow_buttocks_fallback:
            hx = (left_hip[0] + right_hip[0]) / 2
            hy = (left_hip[1] + right_hip[1]) / 2
            polys.append({
                "id": f"person_{pid}_buttocks",
                "person_id": pid,
                "part": "buttocks",
                "points": ellipse_polygon(hx, hy + torso_h * 0.18, torso_w * 0.32, torso_h * 0.18, n=24)
            })

        # Placeholder sensitive zones: geometry-based, not semantic detectors.
        allow_male_groin_fallback = (
            settings.get("male_groin", False)
            and torso_anchor
            and lower_body_anchor
            and pose_box_conf >= 0.88
            and silhouette_conf >= 0.80
            and torso_conf >= 0.10
            and 0.15 <= face_conf <= 0.35
            and 0.05 <= feet_conf <= 0.40
        )

        allow_male_groin_video_fallback = (
            settings.get("male_groin", False)
            and not allow_male_groin_fallback
            and stable_track
            and has_side_padding
            and torso_anchor
            and lower_body_anchor
            and lower_body_visible
            and pose_box_conf >= 0.88
            and silhouette_conf >= 0.84
            and torso_conf >= 0.05
            and 0.10 <= face_conf <= 0.40
            and 0.05 <= feet_conf <= 0.45
            and shoulder_y_norm >= 0.38
            and hip_y_norm >= 0.52
            and pose_box_top_norm >= 0.22
        )

        allow_male_groin_temporal = (
            settings.get("male_groin", False)
            and not allow_male_groin_fallback
            and not allow_male_groin_video_fallback
            and male_groin_recent
            and has_any_padding
            and torso_anchor
            and lower_body_anchor
            and lower_body_visible
            and pose_box_conf >= 0.86
            and silhouette_conf >= 0.35
            and 0.08 <= face_conf <= 0.38
            and feet_conf <= 0.30
            and shoulder_y_norm >= 0.38
            and hip_y_norm >= 0.52
            and pose_box_top_norm >= 0.22
        )

        if allow_male_groin_fallback or allow_male_groin_video_fallback or allow_male_groin_temporal:
            if track_state is not None and now is not None:
                track_state["male_groin_last_seen"] = now
                track_state["male_groin_persist_until"] = now + MALE_GROIN_PERSIST_SECONDS
            polys.append({
                "id": f"person_{pid}_male_groin",
                "person_id": pid,
                "part": "male_groin",
                "points": ellipse_polygon(cx, cy + torso_h * 0.29, torso_w * 0.12, torso_h * 0.09, n=18)
            })

        allow_intimate_front_fallback = (
            settings.get("intimate_front", False)
            and torso_anchor
            and lower_body_anchor
            and pose_box_conf >= 0.90
            and silhouette_conf >= 0.60
            and 0.20 <= feet_conf <= 0.65
            and (face_conf >= 0.45 or (torso_conf >= 0.35 and feet_conf >= 0.45))
        )

        if allow_intimate_front_fallback:
            polys.append({
                "id": f"person_{pid}_intimate_front",
                "person_id": pid,
                "part": "intimate_front",
                "points": ellipse_polygon(cx, cy + torso_h * 0.30, torso_w * 0.14, torso_h * 0.10, n=18)
            })

        allow_intimate_back_fallback = (
            settings.get("intimate_back", False)
            and allow_back_fallback
            and allow_buttocks_fallback
            and silhouette_conf >= 0.90
            and face_conf < 0.20
        )

        if allow_intimate_back_fallback:
            polys.append({
                "id": f"person_{pid}_intimate_back",
                "person_id": pid,
                "part": "intimate_back",
                "points": ellipse_polygon(cx, cy + torso_h * 0.30, torso_w * 0.14, torso_h * 0.10, n=18)
            })

    # WRISTS
    if settings.get("wrists", False) and torso_anchor and upper_body_seg_conf >= 0.20 and pose_box_conf >= 0.85:
        for shoulder_idx, elbow_idx, wrist_idx, side in [
            (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "l"),
            (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"], "r"),
        ]:
            if kp_ok(shoulder_idx, 0.35) and kp_ok(elbow_idx, 0.45) and kp_ok(wrist_idx, 0.45):
                idx = wrist_idx
                wx, wy = kp(idx)
                polys.append({
                    "id": f"person_{pid}_wrist_{side}",
                    "person_id": pid,
                    "part": "wrists",
                    "points": circle_polygon(wx, wy, 12, n=16)
                })

    # HANDS
    if settings.get("hands", False) and torso_anchor and upper_body_seg_conf >= 0.20:
        for shoulder_idx, elbow_idx, wrist_idx, side in [
            (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "l"),
            (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"], "r")
        ]:
            if kp_ok(shoulder_idx, 0.35) and kp_ok(elbow_idx, 0.45) and kp_ok(wrist_idx, 0.45):
                elbow = kp(elbow_idx)
                wrist = kp(wrist_idx)
                tip = extend_segment(elbow, wrist, extension=0.48)
                dist = math.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
                radius = max(14, int(dist * 0.16))
                polys.append({
                    "id": f"person_{pid}_hand_{side}",
                    "person_id": pid,
                    "part": "hands",
                    "points": capsule_polygon(wrist, tip, radius=radius, n=12)
                })

    # FOREARMS
    if settings.get("forearms", False) and torso_anchor and upper_body_seg_conf >= 0.20:
        for shoulder_idx, elbow_idx, wrist_idx, side in [
            (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "l"),
            (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"], "r")
        ]:
            if kp_ok(shoulder_idx, 0.35) and kp_ok(elbow_idx, 0.45) and kp_ok(wrist_idx, 0.45):
                elbow = kp(elbow_idx)
                wrist = kp(wrist_idx)
                dist = math.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
                radius = max(12, int(dist * 0.13))
                polys.append({
                    "id": f"person_{pid}_forearm_{side}",
                    "person_id": pid,
                    "part": "forearms",
                    "points": capsule_polygon(elbow, wrist, radius=radius, n=12)
                })

    # HIPS
    if settings.get("hips", False) and torso_anchor and lower_body_anchor and left_hip is not None and right_hip is not None:
        hx = (left_hip[0] + right_hip[0]) / 2
        hy = (left_hip[1] + right_hip[1]) / 2
        polys.append({
            "id": f"person_{pid}_hips",
            "person_id": pid,
            "part": "hips",
            "points": ellipse_polygon(hx, hy, max(22, torso_w * 0.22), max(18, torso_h * 0.12), n=20)
        })

    # THIGHS
    if settings.get("thighs", False) and torso_anchor and lower_body_anchor:
        for hip_idx, knee_idx, side in [
            (KP["left_hip"], KP["left_knee"], "l"),
            (KP["right_hip"], KP["right_knee"], "r")
        ]:
            if kp_ok(hip_idx, 0.45) and kp_ok(knee_idx, 0.35):
                hip = kp(hip_idx)
                knee = kp(knee_idx)
                dist = math.hypot(knee[0] - hip[0], knee[1] - hip[1])
                radius = max(16, int(dist * 0.15))
                polys.append({
                    "id": f"person_{pid}_thigh_{side}",
                    "person_id": pid,
                    "part": "thighs",
                    "points": capsule_polygon(hip, knee, radius=radius, n=12)
                })

    # KNEES
    if settings.get("knees", False) and torso_anchor and lower_body_anchor:
        for knee_idx, side in [(KP["left_knee"], "l"), (KP["right_knee"], "r")]:
            if kp_ok(knee_idx, 0.35):
                kx, ky = kp(knee_idx)
                polys.append({
                    "id": f"person_{pid}_knee_{side}",
                    "person_id": pid,
                    "part": "knees",
                    "points": circle_polygon(kx, ky, 14, n=16)
                })

    # CALVES
    if settings.get("calves", False) and torso_anchor and lower_body_anchor:
        for knee_idx, ankle_idx, side in [
            (KP["left_knee"], KP["left_ankle"], "l"),
            (KP["right_knee"], KP["right_ankle"], "r")
        ]:
            if kp_ok(knee_idx, 0.35) and kp_ok(ankle_idx, 0.30):
                knee = kp(knee_idx)
                ankle = kp(ankle_idx)
                dist = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
                radius = max(14, int(dist * 0.12))
                polys.append({
                    "id": f"person_{pid}_calf_{side}",
                    "person_id": pid,
                    "part": "calves",
                    "points": capsule_polygon(knee, ankle, radius=radius, n=12)
                })

    # FEET
    if settings.get("feet", False) and torso_anchor and lower_body_anchor:
        for knee_idx, ankle_idx, side in [
            (KP["left_knee"], KP["left_ankle"], "l"),
            (KP["right_knee"], KP["right_ankle"], "r")
        ]:
            if kp_ok(knee_idx, 0.35) and kp_ok(ankle_idx, 0.30):
                knee = kp(knee_idx)
                ankle = kp(ankle_idx)
                foot_tip = extend_segment(knee, ankle, extension=0.34)
                dist = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
                radius = max(12, int(dist * 0.10))
                polys.append({
                    "id": f"person_{pid}_foot_{side}",
                    "person_id": pid,
                    "part": "feet",
                    "points": capsule_polygon(ankle, foot_tip, radius=radius, n=12)
                })

    if settings.get("silhouette", False) and silhouette_conf < seg_min_conf_for_part("silhouette") and torso_anchor and pose_box_conf >= 0.70:
        silhouette_points = pose_silhouette_polygon(kpts, confs, w, h)
        if silhouette_points:
            polys.append({
                "id": f"person_{pid}_silhouette",
                "person_id": pid,
                "part": "silhouette",
                "points": silhouette_points,
            })

    return polys


@app.route("/detect", methods=["POST"])
def detect():
    if "frame" not in request.files:
        return jsonify({"polygons": []})

    settings = parse_settings()

    file = request.files["frame"]
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"polygons": []})

    h, w, _ = frame.shape
    active_bounds = detect_active_bounds(frame)
    now = time.monotonic()

    need_pose = any(settings.get(k, False) for k in [
        "face", "eyes", "shoulders", "torso", "chest", "back", "armpits",
        "navel", "buttocks", "hips", "hands", "wrists", "forearms",
        "male_groin",
        "thighs", "knees", "calves", "feet", "intimate_front", "intimate_back",
        "silhouette"
    ])
    seg_class_ids = requested_seg_class_ids(settings)
    need_seg = bool(seg_class_ids)

    pose_result = None
    seg_result = None

    if need_pose:
        pose_result = pose_model.predict(
            frame,
            imgsz=512,
            conf=0.25,
            max_det=4,
            verbose=False
        )[0]

    if need_seg:
        seg_result = seg_model.predict(
            frame,
            imgsz=640,
            conf=min(SEG_CLASS_MIN_CONF.values()),
            classes=seg_class_ids,
            max_det=20,
            verbose=False
        )[0]

    pose_polygons = []
    seg_polygons = []
    pose_centers = []
    pose_ids = []
    pose_box_confs = []
    pose_boxes_xyxy = []
    seg_support = {}
    seg_candidates = []

    if pose_result is not None and pose_result.keypoints is not None:
        kpts_xy = pose_result.keypoints.xy.cpu().numpy()
        kpts_conf = None
        if pose_result.keypoints.conf is not None:
            kpts_conf = pose_result.keypoints.conf.cpu().numpy()
        if pose_result.boxes is not None and pose_result.boxes.conf is not None:
            pose_box_confs = pose_result.boxes.conf.cpu().numpy().astype(float).tolist()
        if pose_result.boxes is not None and pose_result.boxes.xyxy is not None:
            pose_boxes_xyxy = pose_result.boxes.xyxy.cpu().numpy().astype(float).tolist()
        if len(pose_box_confs) < len(kpts_xy):
            pose_box_confs.extend([0.0] * (len(kpts_xy) - len(pose_box_confs)))
        if len(pose_boxes_xyxy) < len(kpts_xy):
            pose_boxes_xyxy.extend([None] * (len(kpts_xy) - len(pose_boxes_xyxy)))

        for person_i, kpts in enumerate(kpts_xy):
            confs = kpts_conf[person_i] if kpts_conf is not None else None
            c = center_from_kpts(kpts, confs, w, h)
            if c is not None:
                pose_centers.append(c)
            else:
                pose_centers.append((w / 2.0, h / 2.0))

        pose_ids = assign_person_ids(pose_centers)

    if need_seg and seg_result is not None and seg_result.masks is not None:
        seg_polys = seg_result.masks.xy
        seg_classes = []
        seg_confs = []
        if seg_result.boxes is not None and seg_result.boxes.cls is not None:
            seg_classes = seg_result.boxes.cls.cpu().numpy().astype(int).tolist()
        if seg_result.boxes is not None and seg_result.boxes.conf is not None:
            seg_confs = seg_result.boxes.conf.cpu().numpy().astype(float).tolist()
        if len(seg_classes) < len(seg_polys):
            seg_classes.extend([0] * (len(seg_polys) - len(seg_classes)))
        if len(seg_confs) < len(seg_polys):
            seg_confs.extend([0.0] * (len(seg_polys) - len(seg_confs)))

        for i, poly in enumerate(seg_polys):
            class_name = SEG_MODEL_NAMES.get(seg_classes[i], "")
            part_name = seg_class_name_to_server_part(class_name)
            conf = float(seg_confs[i])
            update_seg_support(seg_support, part_name, conf)

        for i, poly in enumerate(seg_polys):
            class_name = SEG_MODEL_NAMES.get(seg_classes[i], "")
            part_name = seg_class_name_to_server_part(class_name)
            conf = float(seg_confs[i])
            if not part_name or conf < seg_min_conf_for_part(part_name):
                continue
            arr = np.array(poly, dtype=np.float32)
            if len(arr) < 3:
                continue
            smoothed = resample_points(poly, target_n=32)
            if not smoothed or len(smoothed) < 3:
                continue

            seg_candidates.append({
                "part": part_name,
                "conf": conf,
                "points": smoothed,
                "center": (float(arr[:, 0].mean()), float(arr[:, 1].mean())),
            })

    if pose_result is not None and pose_result.keypoints is not None:
        kpts_xy = pose_result.keypoints.xy.cpu().numpy()
        kpts_conf = None
        if pose_result.keypoints.conf is not None:
            kpts_conf = pose_result.keypoints.conf.cpu().numpy()

        for person_i, kpts in enumerate(kpts_xy):
            confs = kpts_conf[person_i] if kpts_conf is not None else None
            pid = pose_ids[person_i]
            pose_polygons.extend(
                build_person_polygons(
                    kpts,
                    confs,
                    w,
                    h,
                    pid,
                    settings,
                    seg_support,
                    pose_box_confs[person_i] if person_i < len(pose_box_confs) else 0.0,
                    pose_box_xyxy=pose_boxes_xyxy[person_i] if person_i < len(pose_boxes_xyxy) else None,
                    active_bounds=active_bounds,
                    track_state=TRACKS.get(pid),
                    now=now,
                )
            )

    if seg_candidates:
        seg_centers = [item["center"] for item in seg_candidates]
        if pose_ids and pose_centers:
            matched_ids = match_seg_to_pose(seg_centers, pose_centers, pose_ids)
        else:
            matched_ids = assign_person_ids(seg_centers)

        for item, matched_pid in zip(seg_candidates, matched_ids):
            seg_polygons.append({
                "id": f"person_{matched_pid}_{item['part']}",
                "person_id": matched_pid,
                "part": item["part"],
                "points": item["points"]
            })

    seg_parts_emitted = {item["part"] for item in seg_polygons}
    polygons = [item for item in pose_polygons if item["part"] not in seg_parts_emitted]
    polygons.extend(seg_polygons)

    return jsonify({"polygons": polygons})


if __name__ == "__main__":
    app.run(port=5000)
