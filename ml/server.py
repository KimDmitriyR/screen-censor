from flask import Flask, request, jsonify
import cv2
import json
import math
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

pose_model = YOLO("yolo26n-pose.pt")
seg_model = YOLO("yolo26n-seg.pt")

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


def clamp(v, lo, hi):
    return max(lo, min(int(round(v)), hi))


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

        used.add(best_pid)
        TRACKS[best_pid] = {"center": center, "last_seen": now}
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


def build_person_polygons(kpts, confs, w, h, pid, settings):
    polys = []

    def kp_ok(idx):
        return confs is None or confs[idx] >= 0.25

    def kp(idx):
        return point_xy(kpts, idx, w, h)

    left_sh = kp(KP["left_shoulder"]) if kp_ok(KP["left_shoulder"]) else None
    right_sh = kp(KP["right_shoulder"]) if kp_ok(KP["right_shoulder"]) else None
    left_hip = kp(KP["left_hip"]) if kp_ok(KP["left_hip"]) else None
    right_hip = kp(KP["right_hip"]) if kp_ok(KP["right_hip"]) else None

    torso_center = None
    torso_w = 140
    torso_h = 180

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
    if settings.get("shoulders", False):
        for sh_idx, side in [(KP["left_shoulder"], "l"), (KP["right_shoulder"], "r")]:
            if kp_ok(sh_idx):
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

        if settings.get("back", False):
            polys.append({
                "id": f"person_{pid}_back",
                "person_id": pid,
                "part": "back",
                "points": ellipse_polygon(cx, cy + torso_h * 0.03, torso_w * 0.42, torso_h * 0.24, n=26)
            })

        if settings.get("armpits", False):
            for sh_idx, el_idx, side in [
                (KP["left_shoulder"], KP["left_elbow"], "l"),
                (KP["right_shoulder"], KP["right_elbow"], "r")
            ]:
                if kp_ok(sh_idx) and kp_ok(el_idx):
                    sh = kp(sh_idx)
                    el = kp(el_idx)
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

        if settings.get("buttocks", False) and left_hip is not None and right_hip is not None:
            hx = (left_hip[0] + right_hip[0]) / 2
            hy = (left_hip[1] + right_hip[1]) / 2
            polys.append({
                "id": f"person_{pid}_buttocks",
                "person_id": pid,
                "part": "buttocks",
                "points": ellipse_polygon(hx, hy + torso_h * 0.18, torso_w * 0.32, torso_h * 0.18, n=24)
            })

        # Placeholder sensitive zones: geometry-based, not semantic detectors.
        if settings.get("intimate_front", False):
            polys.append({
                "id": f"person_{pid}_intimate_front",
                "person_id": pid,
                "part": "intimate_front",
                "points": ellipse_polygon(cx, cy + torso_h * 0.30, torso_w * 0.14, torso_h * 0.10, n=18)
            })

        if settings.get("intimate_back", False):
            polys.append({
                "id": f"person_{pid}_intimate_back",
                "person_id": pid,
                "part": "intimate_back",
                "points": ellipse_polygon(cx, cy + torso_h * 0.30, torso_w * 0.14, torso_h * 0.10, n=18)
            })

    # WRISTS
    if settings.get("wrists", False):
        for idx, side in [(KP["left_wrist"], "l"), (KP["right_wrist"], "r")]:
            if kp_ok(idx):
                wx, wy = kp(idx)
                polys.append({
                    "id": f"person_{pid}_wrist_{side}",
                    "person_id": pid,
                    "part": "wrists",
                    "points": circle_polygon(wx, wy, 12, n=16)
                })

    # HANDS
    if settings.get("hands", False):
        for elbow_idx, wrist_idx, side in [
            (KP["left_elbow"], KP["left_wrist"], "l"),
            (KP["right_elbow"], KP["right_wrist"], "r")
        ]:
            if kp_ok(elbow_idx) and kp_ok(wrist_idx):
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
    if settings.get("forearms", False):
        for elbow_idx, wrist_idx, side in [
            (KP["left_elbow"], KP["left_wrist"], "l"),
            (KP["right_elbow"], KP["right_wrist"], "r")
        ]:
            if kp_ok(elbow_idx) and kp_ok(wrist_idx):
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
    if settings.get("hips", False) and left_hip is not None and right_hip is not None:
        hx = (left_hip[0] + right_hip[0]) / 2
        hy = (left_hip[1] + right_hip[1]) / 2
        polys.append({
            "id": f"person_{pid}_hips",
            "person_id": pid,
            "part": "hips",
            "points": ellipse_polygon(hx, hy, max(22, torso_w * 0.22), max(18, torso_h * 0.12), n=20)
        })

    # THIGHS
    if settings.get("thighs", False):
        for hip_idx, knee_idx, side in [
            (KP["left_hip"], KP["left_knee"], "l"),
            (KP["right_hip"], KP["right_knee"], "r")
        ]:
            if kp_ok(hip_idx) and kp_ok(knee_idx):
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
    if settings.get("knees", False):
        for knee_idx, side in [(KP["left_knee"], "l"), (KP["right_knee"], "r")]:
            if kp_ok(knee_idx):
                kx, ky = kp(knee_idx)
                polys.append({
                    "id": f"person_{pid}_knee_{side}",
                    "person_id": pid,
                    "part": "knees",
                    "points": circle_polygon(kx, ky, 14, n=16)
                })

    # CALVES
    if settings.get("calves", False):
        for knee_idx, ankle_idx, side in [
            (KP["left_knee"], KP["left_ankle"], "l"),
            (KP["right_knee"], KP["right_ankle"], "r")
        ]:
            if kp_ok(knee_idx) and kp_ok(ankle_idx):
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
    if settings.get("feet", False):
        for knee_idx, ankle_idx, side in [
            (KP["left_knee"], KP["left_ankle"], "l"),
            (KP["right_knee"], KP["right_ankle"], "r")
        ]:
            if kp_ok(knee_idx) and kp_ok(ankle_idx):
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

    need_pose = any(settings.get(k, False) for k in [
        "face", "eyes", "shoulders", "torso", "chest", "back", "armpits",
        "navel", "buttocks", "hips", "hands", "wrists", "forearms",
        "thighs", "knees", "calves", "feet", "intimate_front", "intimate_back"
    ])
    need_seg = settings.get("silhouette", False)

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
            conf=0.25,
            classes=[0],
            max_det=4,
            verbose=False
        )[0]

    polygons = []
    pose_centers = []
    pose_ids = []

    if pose_result is not None and pose_result.keypoints is not None:
        kpts_xy = pose_result.keypoints.xy.cpu().numpy()
        kpts_conf = None
        if pose_result.keypoints.conf is not None:
            kpts_conf = pose_result.keypoints.conf.cpu().numpy()

        for person_i, kpts in enumerate(kpts_xy):
            confs = kpts_conf[person_i] if kpts_conf is not None else None
            c = center_from_kpts(kpts, confs, w, h)
            if c is not None:
                pose_centers.append(c)
            else:
                pose_centers.append((w / 2.0, h / 2.0))

        pose_ids = assign_person_ids(pose_centers)

        for person_i, kpts in enumerate(kpts_xy):
            confs = kpts_conf[person_i] if kpts_conf is not None else None
            pid = pose_ids[person_i]
            polygons.extend(build_person_polygons(kpts, confs, w, h, pid, settings))

    if need_seg and seg_result is not None and seg_result.masks is not None:
        seg_polys = seg_result.masks.xy
        seg_centers = []

        for poly in seg_polys:
            arr = np.array(poly, dtype=np.float32)
            if len(arr) < 3:
                seg_centers.append((w / 2.0, h / 2.0))
                continue
            seg_centers.append((float(arr[:, 0].mean()), float(arr[:, 1].mean())))

        if pose_ids and pose_centers:
            matched_ids = match_seg_to_pose(seg_centers, pose_centers, pose_ids)
        else:
            matched_ids = assign_person_ids(seg_centers)

        for i, poly in enumerate(seg_polys):
            smoothed = resample_points(poly, target_n=32)
            if smoothed and len(smoothed) >= 3:
                polygons.append({
                    "id": f"person_{matched_ids[i]}_silhouette",
                    "person_id": matched_ids[i],
                    "part": "silhouette",
                    "points": smoothed
                })

    return jsonify({"polygons": polygons})


if __name__ == "__main__":
    app.run(port=5000)